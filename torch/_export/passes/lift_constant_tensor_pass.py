import torch
from torch._guards import detect_fake_mode
from torch.export.exported_program import InputKind, InputSpec, TensorArgument


def lift_constant_tensor_pass(gm, graph_signature, state_dict):
    """
    Takes an ExportedProgram and returns the ExportedProgram modified in-place,
    with the constant tensors as buffers.
    """
    if len([node for node in gm.graph.nodes if node.op == "placeholder"]) == 0:
        return

    buffers = graph_signature.buffers

    fake_mode = detect_fake_mode(
        tuple(node.meta["val"] for node in gm.graph.nodes if node.op == "placeholder")
    )
    assert fake_mode is not None

    first_user_input = None
    lifted_buffers = []
    for node in gm.graph.nodes:
        if node.op == "placeholder" and node.name in graph_signature.user_inputs:
            first_user_input = node
            break

    for node in gm.graph.nodes:
        if node.op == "get_attr":
            constant_tensor = getattr(gm, node.target)
            if not isinstance(constant_tensor, torch.Tensor):
                continue

            constant_tensor_fqn = f"_lifted_tensor_constant{len(buffers)}"

            with gm.graph.inserting_before(first_user_input):
                # Insert the constant node before the first user input
                const_placeholder_node = gm.graph.placeholder(constant_tensor_fqn)
                for k, v in node.meta.items():
                    const_placeholder_node.meta[k] = v
                const_placeholder_node.meta["val"] = fake_mode.from_tensor(
                    constant_tensor, static_shapes=True
                )
                const_placeholder_node.meta["val"].constant = constant_tensor
                node.replace_all_uses_with(const_placeholder_node)
                gm.graph.erase_node(node)

                # Add the constant as a buffer to the graph signature
                lifted_buffers.append(
                    InputSpec(
                        kind=InputKind.BUFFER,
                        arg=TensorArgument(name=const_placeholder_node.name),
                        target=constant_tensor_fqn,
                    )
                )
                buffers.append(constant_tensor_fqn)
                state_dict[constant_tensor_fqn] = constant_tensor

    new_input_specs = []
    for s in graph_signature.input_specs:
        if s.kind == InputKind.USER_INPUT and len(lifted_buffers) > 0:
            new_input_specs.extend(lifted_buffers)
            lifted_buffers.clear()
        new_input_specs.append(s)
    graph_signature.input_specs = new_input_specs
    gm.recompile()
