import os
import sys
import subprocess
import torch

sys.path.append('../../')


def call_file_main():
    # Define the command and its arguments as separate items in a list
    command = ["python", "main.py", "--model", "MNAD", "--method", "MNAD", "--phase", "train", "--dataset", "ped2",
               "--device", "cuda:0"]

    # Use subprocess to run the script
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running {command}: {e}")


def call_file_keyframes():
    # Define the path to the Python script you want to run
    # script_file = os.path.join('dataset', 'KF_DE_Entropy_ped2.py')
    script_file = os.path.join('../dataset', 'KF_OF.py')
    command = ["python", script_file]
    # Use subprocess to run the script
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running {command}: {e}")


def test():
    # Create a sample tensor
    tensor = torch.tensor([[1, 2],
                           [3, 4]], dtype=torch.float32)
    print('tensor.shape = ', tensor.shape)

    # Resize the tensor to a new shape (e.g., 3x3)
    new_shape = (3, 3)
    resized_tensor = torch.nn.functional.interpolate(tensor,
                                                     size=new_shape, mode="bilinear",
                                                     align_corners=False)
    print('resized_tensor.shape = ', resized_tensor.shape)

    # Print the resized tensor
    print(resized_tensor)


def main():
    # call_file_main()
    call_file_keyframes()
    # test()


if __name__ == '__main__':
    main()
