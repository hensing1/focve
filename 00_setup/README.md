# Setup Instructions and PyTorch Introduction

Welcome to **Foundations of 4D/6D Object Capture for Virtual Environments**! 
In the upcoming 3 weeks, you will learn a lot about the latest techniques and trends in 3D/4D/6D visual data processing. 
The exercises will deal with a large variety of topics and will require you to setup both Unity and a python environment.
To facilitate the exercises for you and give you more time to make sure, everything can run smoohtly, we have prepared this document that includes

- some general hints on the technical necessities that are required for the exercises,
- a guide on how to set up Unity on your own PC, or how to use the Unity installation we provide for you on the pool computers, and a test project to see if it works,
- instructions for building the python environment both on your PC and on the pool computers, and
- an initial exercise sheet for you to get to know PyTorch and the workflow for the python exercises (just for preparation, it does not count towards exam admission).

**Please read the instructions on this sheet carefully!** Even if you have already worked with Unity or Python/PyTorch, please read the instructions in this document. Especially working with the pool computers requires you to consider some particularities. It would be best if you have already tried to set up Unity **before the first exercise** so everything runs smoothly. If you have questions or problems, you can send us an email or ask us when the course and the exercises start.


## Preparation for the Exercises

For the exercises, you may use your own PCs, or the computers we provide, which will already have some packages and programs preinstalled. 

If you want to use **your own computer**:

- Make sure you have **free space** left on your device! Both Unity and PyTorch require some free memory to be installed.
- Make sure you have access to an adequate **NVIDIA GPU**, as later exercises will be very computation heavy.

If you want to use the **pool computers** with our provided setup:

- Make sure that you have a **working [account](https://gsg.cs.uni-bonn.de/doku.php?id=en:accounts:students:manage)**.
- Make sure you have **free space** so that your quota is not exceeded when you download the exercises.
- You can check your used space by typing `quota` in a terminal.
- We recommend that you mainly work in the `/tmp` directory of the pool computers to avoid issues with your quota. Note the information in the next section!

### Pool Computers and Storage Space
We generally recommend that you work in the `/tmp` directory of the pool computers, as it offers more storage capacity.
Create a folder to work in within the `/tmp` directory of the computer. Name your folder e.g. after your account name, and add anything you need to that folder. 

The `/tmp` folder is a local directory on each pool computer that is **not synchronized**, and also **deleted every night**. Anything you want to keep after working in there, you should **back up**. The computers usually shut down around 8 PM.

Note that, to download the exercises, you still need at least some free space left in your quota, as your downloads are part of it. You can move anything to your folder in the `/tmp` directory afterwards.

## Setting up Unity - On Your Own Computer

### Installing Unity Hub
If you have not worked with Unity before, you will have to [install Unity Hub](https://docs.unity3d.com/hub/manual/InstallHub.html) on your machine. In Unity Hub, you will have to log in or create a free account, so you have a valid license. 

![unity hub sign in](./assets/unity_hub.png)

Agree to the *Editor Software Terms* for the Unity Personal license when prompted.

### Installing the Editor 
Afterwards, within Unity Hub, you will be able to install editors. 
We suggest you do that **by opening a project** (see next section) and installing the correct version when prompted.
The Unity Editor version of our projects is 6000.3.4f1. 

You can see any installed Editors in the *Installs* tab. Initially, no editors will be installed

![installs tab](./assets/unity_hub_installs_tab.png)

In this tab, you can also install other versions, or locate existing versions on your PC to add them manually.
If you want to install a specific version, you can also look [here](https://unity.com/releases/editor/archive) to install it and add it to the Hub manually. 


### Opening a Project
To open a project, in the *Projects* tab select *Add*, select *Add project from disk* and select the folder of the exercise you want to work on. For the test project, choose the folder `unity_testproject`. (For the exercises, this will simply be the folder of the respective exercise.)

![add project](./assets/unity_hub_add_project.png)

Unity gives you a *Missing Editor Version warning* if you have not installed the Editor the project was created with. You can install the corresponding editor now. 

![missing editor warning](./assets/unity_hub_missing_editor.png)

You only need to install the editor, no additional modules.

![missing editor install](./assets/unity_hub_missing_editor_install.png)

The installation may take a moment, you can see the progress in the top bar or Unity Hub.

You can open an added project by clicking on its name in the *Projects* tab.

![projects tab](./assets/unity_hub_projects_tab.png)

Accept the *Unity Editor Software Terms* if prompted.
![software_terms](./assets/unity_editor_software_terms.png)

If you open a project for the first time, it will take a long time to load. Depending on your machine, this can take several minutes. Usually, a loading bar appears, but it might not be shown. So give it some time.
If you have waited several minutes and nothing happened, maybe check your task manager or click on the project again.

If you want to open the project with a different editor version than the one it was created with, Unity prompts you to change the version of the editor for the project (see below).

### Changing the Editor Version of a Project

We recommend using the Editor version that the project was created with, see above.
If for some reason you want to use a different version of the editor than the one a project was created with, you may choose it from the list that appears when opening a project.
This will prompt a series of pop ups. Click through them to open the project with the selected editor version.

## Setting up Unity - Using the Pool Computers

The pool computers already have an installation of the Unity Hub and a corresponding editor. **Do NOT install anything additionally!** 

We **strongly recommend** that you work on the Unity exercises in the `/tmp` directory, as it does not impact your quota. Unity will require additional memory for any project you open in its respective folder.
Remember to **save anything that you want to keep** after working on the exercises as the contents of `/tmp` get deleted every night.

### Opening Unity Hub

The installation of Unity Hub is located in `/opt/unityhub`. You can start Unity Hub e.g. via the Applications tab via `Applications -> Development -> Unity Hub`. 
Create an account if you don't already have one, since you will need a working license to start the editor. 

![unity hub sign in](./assets/unity_hub.png)

### Finding the Editor Installation

In the *Installs* tab, you can see a list of all installed editors, which should be empty at first.

Add the preinstalled Unity Editor, which is located under `/opt/unityeditor/6000.3.4f1/Editor/Unity`, by clicking on `Locate` and finding the installation.

![installs tab](./assets/pools_unity_hub_installs_tab_empty.png)
![locate editor](./assets/pools_unity_hub_locate_editor.png)
![locate editor 2](./assets/pools_unity_hub_locate_editor_2.png)
![locate editor 3](./assets/pools_unity_hub_locate_editor_3.png)

The installation should appear in the list.

![installs tab](./assets/pools_unity_hub_installs_tab.png)


### Opening a Project

Afterwards, you need to add the exercise project.
Navigate to the *Projects* tab and add a project by clickung on `Add -> Add project from disk`.

![add project](./assets/pools_unity_hub_add_project.png)

Find the project in the exercise, in this case the directory `unity_testproject`. (In later exercises, this will simply be the exercise folder.)

![find project](./assets/pools_unity_hub_add_project_2.png)

Unity Hub should recognize by itself that you have the correct editor installed.
Your added project is now visible in the *Projects* tab, you can open it by clicking on it. 

![projects tab](./assets/pools_unity_hub_projects_tab.png)

Accept the *Unity Editor Software Terms* when prompted.
![software_terms](./assets/unity_editor_software_terms.png)

The first time you open a project it will take a long time to load. This can take several minutes. On the pool computers, the loading bar sometimes is not shown, so really give it some time. If you have waited several minutes and nothing happened, maybe check your task manager or click on the project again.

Generally, Unity can be used via remote desktop. However, the demos might not run smoothly.

## Working with the Unity Editor

This is the layout of the Unity editor.

![editor overview](./assets/unity_editor.png)

In the center, you see the *Scene* view, where you can select and manipulate the placement of *GameObjects* manually, as well as the *Game* view, which displays the image that will be visible to the player according to the camera in the scene. Note that you cannot manipulate objects via the *Game* view in case you accidentally select the tab.

In the *Project* view you can explore the assets that are part of the project. We have grouped them into the folders `Materials`, `Scenes`, `Scripts`, `Settings` and `Tests`.
To open a scene, select it within the `Scenes` directory.

![scene opened](./assets/unity_editor_scene_opened.png)

The *Console* view is hidden as a tab behind the *Project* view. Debug output including error messages and most importantly compiler errors are logged in the *Console* view.
The *Hierarchy* view shows the hierarchy of *GameObjects* in the current scene. 
The *Inspector* view shows the properties of the selected object in the *Hierarchy* or *Project* view.

The first sheet will include some instructions on how to code with Unity in the context of our lecture.
Writing the code will be performed in a seperate editor, e.g. Visual Studio Code, which is preinstalled on the pool computers.

### Running the Demo

To run the demo, click on the play button in at the top.

![run demo](./assets/unity_editor_run_demo.png)

The demo will look something like this.

![demo screen](./assets/unity_editor_demo_test.png)

### Running the tests

To run the tests in Unity, you first need to open the test runner by selecting `Window -> General -> Test Runner`.

![open test runner](./assets/UnityEditorTestRunnerMenu.png)

It will open another window within Unity.
If the window is too small and does not show the *Run All* button, resize it.

![resize test runner window](./assets/unity_editor_test_runner_small.png)
![resize test runner window 2](./assets/unity_editor_test_runner_big.png)

If all tests run smoothly, you will see a green checkmark next to the tests.

![test success](./assets/unity_editor_test_runner_success.png)

You cannot run the tests while the demo is running.

## Building the Python Environment - On your own computer

With every Python exercise, we provide a `requirements.txt` that lists all relevant packages for the corresponding sheet. 
We suggest using conda (e.g. with [miniforge](https://github.com/conda-forge/miniforge)) to create a separate environment for this lecture so you do not interfere with any of your other installations.
In the case of version mismatches, please try out different combinations or ask your tutor in the exercise.
The voluntary exercise sheet that is part of this initial material will teach you how to run the tests using Nox.

## Building the Python Environment - Using the Pool Computers

Due to some python packages requiring a lot of storage, we made sure that those are preinstalled on the pool computers. **Do NOT install all packages listed in the requirements file!** If you do, your storage most likely will run full, you have to uninstall everything again, and start from scratch.

**Instead** simply install the packages that are missing individually one after another via `pip install <package>`. The list of installed packages can be accessed via `pip list`. The packages required for each exercise are listed in `requirements.txt`.
Note that anything you install will only be installed for your user (counting towards your quota), and you cannot change the preinstalled versions.
You can however install the same packages again, which uses unnecessary memory, so it is better if you work with the preinstalled packages.
In the case of version mismatches, please try out different combinations or ask your tutor in the exercise.
The voluntary exercise sheet that is part of this initial material will teach you how to run the tests using Nox.

If you want to use the pool computers remotely: Running the tests works fine. However, most of the demos cannot be run remotely, as they cannot be started with only a terminal or using a remote desktop application.

## Voluntary PyTorch Exercise - Tensor Operations and Broadcasting

Over the course of the next exercise sheets, you will need to work with PyTorch using tensors. The aim of this exercise sheet is for you to get to know the basics of tensor operations, especially regarding broadcasting.

**This exercise sheet is completely voluntary. It will not be graded or count towards exam admission.** You may solve the whole sheet, only work on the parts that are interesting to you, or skip it entirely.

Hints are provided to help you solve the exercises, or just to give you a general idea of how to work with tensors. Even in the case that you do not want to solve the exercises, these hints might still be helpful to you.

At the end of the exercise sheet you will find additional remarks regarding solving this exercise sheet.


### Task

In `src/tensors.py`, complete the following exercises:

1. **Tensor Information**: Implement the function `get_info` to return the number of dimensions, the data type, the shape and the device of a given tensor.

    **Hint**: Tensors have different properties that are helpful to know, e.g. for debugging purposes. Mismatches in the [number of dimensions](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.ndim.html), the [data type](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.torch.dtype) and the [shape](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.shape.html) of tensors can easily lead to errors when running the code. Furthermore, as PyTorch can utilize GPUs for parallel computations, two tensors need to be stored on the same [device](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.device.html) (CPU or one of the available GPUs) if you want to perform an operation using both.
2. **Slicing**: Implement the function`swap_matrix_quadrant` that, given a tensor of shape 4x4, swaps its quadrants of size 2x2 as follows, and returns the result as a new tensor, without modifying the input tensor.

    ```
        A = ⌈ A11 A12 ⌉ -> ⌈ A22 A12 ⌉
            ⌊ A21 A22 ⌋    ⌊ A21 A11 ⌋
    ```
    **Hint**: To access the `i`-th element of a tensor `T` in its first dimension, one uses `T[i]`. Negative indices reference the tensor from back to front. To get a slice instead of a single element, one uses `T[start:end:step]`, where omitting one of the values uses the defaults 0, the size in the indexed dimension, and 1. A negative `step` swaps the order of the returned values.

    **Hint**: To make an explicit copy of a tensor, use [torch.Tensor.clone](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.clone.html). Simply assigning a tensor to a new variable and modifying any of the two will also modify the other tensor as they reference the same memory, which can lead to unwanted behavior.
3. **Elementary and Elementwise Operations**: Implement the function `normalize_and_abs` that first normalizes some values in a tensor given a minimum and a maximum to `[-1, 1]`, and then computes the absolute value for each normalized entry and returns it.

    **Hint**: PyTorch supports elementary operations such as `+`, `-`, `*`, `/` etc. between two tensors of the same size or via broadcasting, and also between a tensor and a scalar value, which are applied as elementwise operations. Other elementwise operations such as [abs](https://docs.pytorch.org/docs/stable/generated/torch.abs.html), [cos](https://docs.pytorch.org/docs/stable/generated/torch.cos.html), [exp](https://docs.pytorch.org/docs/stable/generated/torch.exp.html) etc. can also applied to a single tensor without changing the tensor's shape.
4. **Boolean Array Indexing**: Implement the function `replace_near_zero` that replaces any tensor entries of a given tensor, that are in the interval [-1, 1], with 0.
    
    **Hint**: Tensors containing booleans (masks) can be utilized to index only elements of a tensor that fulfil certain conditions. Passing a mask as an index to another tensor selects the elements that are `True` in the mask. Both tensors need to have the same shape. A mask can be created e.g. by comparing a tensor to a scalar.

    **Hint**: For elementwise boolean operations e.g. [torch.logical_and](https://docs.pytorch.org/docs/stable/generated/torch.logical_and.html) and [torch.logical_or](https://docs.pytorch.org/docs/stable/generated/torch.logical_or.html) can be used. [torch.logical_not](https://docs.pytorch.org/docs/stable/generated/torch.logical_not.html) or `~` inverts a boolean tensor.
5. **Integer Array Indexing**: Implement the function `select_matrices_from_batch` that receives a batch of 2D matrices and an array of indices, and returns a tensor with only the 2D matrices selected by the indices.

    **Hint**: To select a new tensor from a given tensor that only contains entries at specific position one can pass an integer array as index. Tensors with multiple dimensions require separate indices per dimension, where the indexing tensors need to be broadcastable and can only contain indices within the bounds of the tensor. Different indexing methods (slicing, masking, ...) can be combined.

    Example:

    ```
    x = torch.tensor([10, 20, 30, 40, 50])        x[[1, 3]] -> [20, 40]
    y = torch.tensor([[1, 2], [3, 4], [5, 6]])    y[[0, 2], [1, 0]] -> [2, 5]
    ```
6. **Tensor Generation and Data Type and Device Conversion**: Implement the function `generate_and_convert_tensors` that performs the following set of tasks:
    1. Create a tensor from the given numpy array
    2. Create a tensor containing only zeros with the same shape as the tensor created from the numpy array
    3. Create a tensor of size 4x4 that is filled with the integer value 42
    4. Convert the tensor created from the numpy array to 8-bit unsigned integers
    5. Move the 8-bit unsigned integer tensor to the given device

    **Hint**: There are several option to create new tensors, including 
    - [torch.from_numpy](https://docs.pytorch.org/docs/stable/generated/torch.from_numpy.html): generate a tensor from a given numpy array, the memory is shared
    - [torch.tensor](https://docs.pytorch.org/docs/stable/generated/torch.tensor.html): generate an independent tensor from a numpy array, a list or a tuple
    - [torch.rand](https://docs.pytorch.org/docs/stable/generated/torch.rand.html): generate a tensor of given size with uniformly distributed random values between 0 and 1
    - [torch.zeros](https://docs.pytorch.org/docs/stable/generated/torch.zeros.html): generate a tensor full of zeros with given size
    - [torch.zeros_like](https://docs.pytorch.org/docs/stable/generated/torch.zeros_like.html): generate a tensor full of zeros with the size of the given tensor
    - [torch.ones](https://docs.pytorch.org/docs/stable/generated/torch.ones.html): generate a tensor full of ones with given size
    - [torch.full](https://docs.pytorch.org/docs/stable/generated/torch.full.html): generate a tensor of given size full of the given value

    **Hint**: Sometimes, a different data type is required for a tensor, e.g. `torch.float32` instead of `torch.uint8` or it needs to be moved to a different device so it can be used. This is performed with [torch.Tensor.to](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.to.html) in both cases.
7. **Dimensions as Arguments**: Implement the function `max_column_sum` that computes the sum of all the values in each column of a 2D tensor and then returns the maximum of the sums as a 1D tensor (not a scalar).

    **Hint**: Some functions accept the dimension as an optional argument `dim`, so that an operation is only performed along that axis, e.g. [torch.sum](https://docs.pytorch.org/docs/stable/generated/torch.sum.html), [torch.min](https://docs.pytorch.org/docs/stable/generated/torch.min.html), [torch.amin](https://docs.pytorch.org/docs/stable/generated/torch.amin.html), [torch.max](https://docs.pytorch.org/docs/stable/generated/torch.max.html) or [torch.amax](https://docs.pytorch.org/docs/stable/generated/torch.amax.html). It is possible to pass a list or tuple to perform the operation across multiple dimensions. The boolean argument `keepdim` determines if the operation should reduce the number of dimensions or if the corresponding dimensions should be kept.
    **Hint**: The _first column_ means

    ```
        [[1,2,3],        [[1],
    A =  [4,5,6],   ->    [4],
         [7,8,9]]         [7]]
    ````
8. **Concatenation and Stacking**: Implement the function `create_matrix_from_vectors` that creates a matrix from three given vectors, that are used as the columns of the matrix in the order they are provided to the function.

    **Hint**: To merge several tensors, you have two options:
    - You can use [torch.cat](https://docs.pytorch.org/docs/stable/generated/torch.cat.html) to concatenate them along an existing axis. In this case, the shape of the tensors must be the same, apart from the dimension used for concatenation.
    - You can use [torch.stack](https://docs.pytorch.org/docs/stable/generated/torch.stack.html) to stack them along a new axis. In this case, all tensors must have the exact same shape, and the new dimension is added at the given position (default is 0).
9. **View and Reshape**: Implement the function `interleave_vectors` that takes two 1D vectors of the same length and returns a 1D vector that contains the entries of both tensors in an alternating order as follows:

    ```
    x = [x1, x2, ..., xn], y = [y1, y2, ..., yn] -> [x1, y1, x2, y2, ..., xn, yn]
    ```
    **Hint**: To represent a tensor in a different shape, [torch.Tensor.view](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.view.html) returns a "view" of a tensor with the desired shape, which references the same memory as the input tensor. [torch.reshape](https://docs.pytorch.org/docs/stable/generated/torch.reshape.html) returns a view of the tensor if possible, else it returns a copy of the tensor. In both cases, the original and new shape need to have the same size (number of elements). Setting a single dimension to `-1` infers its size from the remaining dimensions and the size of the input tensor.
10. **Transposing**: Implement the function `transpose_matrices` that receives either a 2D matrix or a 3D tensor containing a batch of 2D matrices and transposes the 2D matrix or all 2D matrices, respectively.

    **Hint**: Using [torch.transpose](https://docs.pytorch.org/docs/stable/generated/torch.transpose.html), any two dimensions of a tensor can be transposed, while [torch.t](https://docs.pytorch.org/docs/stable/generated/torch.t.html) only works on 2D tensors, and leaves 0D and 1D tensors unchanged.
11. **Broadcasting and Singleton Dimensions**:

    (a) Implement the functions `make_broadcastable_1`, `make_broadcastable_2`, and `make_broadcastable_3` that get two tensors as input and adjust the dimensions of the tensors in a way that they are broadcastable when returned and can be added.

    (b) Implement the function `batch_scalar_product` that computes the scalar products of pairs of vectors from two batches (for the `i`-th scalar product, vector `i` in batch 1 and vector `i` in batch two are used). Use [torch.bmm](https://docs.pytorch.org/docs/stable/generated/torch.bmm.html) for the computation. 

    **Hint**: Broadcasting follows a [set of rules](https://docs.pytorch.org/docs/stable/notes/broadcasting.html) that determines, if two tensors are broadcastable (automatically expanded to have the same size). These are:
    - Each tensor needs to have at least one dimension.
    - When iterating over the dimension sizes starting at the last dimension, the dimension sizes must either
        * be equal,
        * one of them is 1, or
        * one of them does not exist.

    As an example, a tensor of size `torch.Size([5,1,4,1])` and a second tensor of size `torch.Size([3,1,1])` are broadcastable:
    - The last dimension is equal (1),
    - the second to last dimension is 4 and 1 in the respective tensors, of which one is 1,
    - the third to last dimension is 1 and 3 in the respective tensors, of which one is 1, and
    - the forth to last dimension (which is the first) does not exist for the second tensor.

    The resulting tensor would be of shape `torch.Size([5,3,4,1])`.

    **Hint**: Sometimes, a singleton dimension (of size 1) needs to be added or deleted to use broadcasting with two tensors.
    - [torch.unsqueeze](https://docs.pytorch.org/docs/stable/generated/torch.unsqueeze.html) adds a new dimension at the given position. Indexing a dimension of a tensor as `None` does the same. Furthermore, [torch.Tensor.expand](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.expand.html) and [torch.broadcast_to](https://docs.pytorch.org/docs/stable/generated/torch.broadcast_to.html) can be used to broadcast a tensor to a given shape and return a view of the tensor. If increasing the number of dimensions like this, it will add new dimensions in the front. [torch.reshape](https://docs.pytorch.org/docs/stable/generated/torch.reshape.html) and [torch.Tensor.view](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.view.html) are additional options to add dimensions.
    - [torch.squeeze](https://docs.pytorch.org/docs/stable/generated/torch.squeeze.html) removes a dimension at a given position, if the size of the dimension is 1, or removes all dimensions of size 1 without a given dimension. Alternatively, indexing the respective dimension with 0, as well as [torch.reshape](https://docs.pytorch.org/docs/stable/generated/torch.reshape.html) and [torch.Tensor.view](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.view.html) can be used.

    **Hint**: Other options to multiply two matrices are [torch.matmul](https://docs.pytorch.org/docs/stable/generated/torch.matmul.html) or the `@`-operator, which offer more options on how two matrices can be combined, depending on their dimensionalities. [torch.bmm](https://docs.pytorch.org/docs/stable/generated/torch.bmm.html) requires both matrices to be 3-dimensional.


### General Remarks

This exercise sheet is voluntary and will not be graded. A set of unit tests is provided for you to check your solutions. To run them, use 

```
nox -s tests
```

In the case that you only want to solve a subset of the tasks from this exercise sheet, you may adjust the variable `tasks_solved` in `tests/test_tensors.py`: Deleting an exercise number from the list will also exclude the corresponding tests when you run the test. This way, the tests for exercises you did not solve will not fail.

After completing the tasks, you should see the result in `demo_output.md` when running the `demo.py` example. In the `demo.py` file, you will also find the `tasks_solved` variable. If you are interested in only the outputs of a subset of exercises, you may also remove numbers of unsolved tasks to only generate the output relevant to you.

As an additional remark: When solving the exercises and in general when working with PyTorch, it can be very helpful to know the shapes of the tensors you are working with (especially when revisiting code after some time or for debugging), e.g. as comments after any operations applied to the tensors:

```python
A = torch.rand(1000,3) # batch_dim x 3
```

<br/>
<center><h3>Good Luck!</h3></center>
