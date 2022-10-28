from openmmml.mlpotential import MLPotential, MLPotentialImpl, MLPotentialImplFactory
import openmm
from typing import Optional, Iterable


class MACEPotentialImplFactory(MLPotentialImplFactory):
    """
    A factory to create ACEPotential objects.
    """

    def createImpl(self, name: str, **args) -> MLPotentialImpl:
        return MACEPotentialImpl(name, **args)


class MACEPotentialImpl(MLPotentialImpl):
    """
    This is the MLPotentialImpl implementing the MACE model built using the pytorch version of MACE.


    MACE does not have a central location to distribute models yet and so we assume the model is in a local
    file and must be passed to the init.
    """

    def __init__(self, name: str, model_path: str):
        self.name = name
        self.model_path = model_path

    def addForces(self,
                  topology: openmm.app.Topology,
                  system: openmm.System,
                  atoms: Optional[Iterable[int]],
                  forceGroup: int,
                  filename: str = "macemodel.pt",
                  **args):
        """Create the mace model"""
        import torch
        import openmmtorch
        from mace.data.utils import Configuration, AtomicNumberTable
        from mace.tools.torch_geometric import DataLoader
        from mace.data import AtomicData
        from e3nn.util import jit
        import numpy as np
        from torch_nl import compute_neighborlist

        torch.set_default_dtype(torch.float64)

        # load the mace model
        model = torch.load(self.model_path)
        z_table = AtomicNumberTable([int(z) for z in model.atomic_numbers])
        # find the atomic numbers
        target_atoms = list(topology.atoms())
        if atoms is not None:
            target_atoms = [target_atoms[i] for i in atoms]
        atomic_numbers = np.array([atom.element.atomic_number for atom in target_atoms])
        is_periodic = (topology.getPeriodicBoxVectors() is not None) or system.usesPeriodicBoundaryConditions()

        if is_periodic:
            pbc = (True, True, True)
        else:
            pbc = (False, False, False)

        config = Configuration(
            atomic_numbers=atomic_numbers,
            positions=np.zeros((len(atomic_numbers), 3)),
            pbc=pbc,
            cell=np.zeros((3, 3)),
            weight=1
        )
        data_loader = DataLoader(dataset=[AtomicData.from_config(config, z_table=z_table, cutoff=model.r_max)],
            batch_size=1,
            shuffle=False,
            drop_last=False,
        )
        input_dict = next(iter(data_loader)).to_dict()
        input_dict.pop("edge_index")
        input_dict.pop("energy", None)
        input_dict.pop("forces", None)
        input_dict.pop("positions")
        input_dict.pop("shifts")
        input_dict.pop("weight")


        class MACEForce(torch.nn.Module):
            """A wrapper around a MACe model which can be called by openmm-torch"""

            def __init__(self, model, input_data, atoms, periodic):
                """
                Args:
                    model:
                        The MACE model which should be wrapped by the class
                    node_attrs:
                        A tensor of the one hot encoded atom nodes
                    atoms:
                        The indices of the atoms in the openmm topology object, used to extract the correct positions
                    periodic:
                        If the system cell is periodic or not, used to calculate the atomic distances
                """

                super(MACEForce, self).__init__()
                self.model = model
                self.input_dict = input_data
                if atoms is None:
                    self.indices = None
                else:
                    self.indices = torch.tensor(sorted(atoms), dtype=torch.int64)
                if periodic:
                    self.pbc = torch.tensor([True, True, True], dtype=torch.bool, requires_grad=False)
                else:
                    self.pbc = torch.tensor([False, False, False], dtype=torch.bool, requires_grad=False)

            def forward(self, positions, boxvectors: Optional[torch.Tensor] = None):
                """
                Evaluate the mace model on the selected atoms.

                Args:
                    positions: torch.Tensor shape (nparticles, 3)
                        The positions of all atoms in the system in nanometers.
                    boxvectors: torch.Tensor shape (3,3)
                        The box vectors of the periodic cell in nanometers

                Returns:
                    energy: torch.Scalar
                        The potential energy in KJ/mol
                    forces: torch.Tensor shape (nparticles, 3)
                        The forces on each atom in (KJ/mol/nm)
                """
                # MACE expects inputs in Angstroms
                # create a config for the model
                positions = positions.to(torch.float64)
                # if we are only modeling a subsection select those positions
                if self.indices is not None:
                    positions = positions[self.indices]
                positions = 10.0*positions

                if boxvectors is None:
                    cell = torch.tensor([[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 100.0]], requires_grad=False, dtype=torch.float64)
                else:
                    boxvectors = boxvectors.to(torch.float64)
                    cell = 10.0*boxvectors

                # pass through the model
                mapping, batch_mapping, shifts_idx = compute_neighborlist(
                    cutoff=self.model.r_max,
                    pos=positions,
                    cell=cell,
                    pbc=self.pbc,
                    batch=torch.zeros(positions.shape[0], dtype=torch.long),
                    self_interaction=True)

                # Eliminate self-edges that don't cross periodic boundaries
                true_self_edge = mapping[0] == mapping[1]
                true_self_edge &= torch.all(shifts_idx == 0, dim=1)
                keep_edge = ~true_self_edge

                # Note: after eliminating self-edges, it can be that no edges remain in this system
                sender = mapping[0][keep_edge]
                receiver = mapping[1][keep_edge]
                shifts_idx = shifts_idx[keep_edge]

                edge_index = torch.stack((sender, receiver))
                inp_dict_this_config = self.input_dict.copy()
                inp_dict_this_config["positions"] = positions
                inp_dict_this_config["edge_index"] = edge_index
                inp_dict_this_config["shifts"] = shifts_idx
                inp_dict_this_config["cell"] = cell

                res = self.model(inp_dict_this_config, compute_force=True)
                return 96.486*res["energy"]


        mace_force = MACEForce(model=model, input_data=input_dict, atoms=atoms, periodic=is_periodic)

        # convert using jit
        module = jit.script(mace_force)
        module.save(filename)

        # Create the openmm torch force

        force = openmmtorch.TorchForce(filename)
        force.setForceGroup(forceGroup)
        force.setUsesPeriodicBoundaryConditions(is_periodic)
        # force.setOutputsForces(True)
        system.addForce(force)


MLPotential.registerImplFactory("mace", MACEPotentialImplFactory())
