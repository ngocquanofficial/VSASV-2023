import torch

def process_tensor(input_tensor):
    desired_shape = (128, 124)

    if input_tensor.size(1) > desired_shape[1]:
        processed_tensor = input_tensor[:, :desired_shape[1]]
    else:
        mean_values = input_tensor.mean(dim=1, keepdim=True)
        fill_values = mean_values.expand(-1, desired_shape[1] - input_tensor.size(1))
        processed_tensor = torch.cat([input_tensor, fill_values], dim=1)

    return processed_tensor

# Example usage:
# Assuming 'your_tensor' is your original tensor with shape (x, y)
your_tensor = torch.rand(125, 124)  # Replace this with your actual tensor
result_tensor = process_tensor(your_tensor)
print(result_tensor.shape)