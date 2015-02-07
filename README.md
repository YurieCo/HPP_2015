# HPP_2015

This repository contains the solutions and projects needed to build and run machine problems for the Heterogeneous Parallel Programming course offered on Coursera for 2015.  These
solutions and projects will built using the professional version of Visual Studio 2010, 2012, and 2013.  Express editions of these Visual Studio versions might work, although
might require editing of the solution and/or projecct files.  

LOCAL DEVELOPMENT IS NOT A REPLACEMENT FOR WEBGPU.  You must still use WebGPU for testing and submitting your code.

### Downloading Visual Studio
If you do not currently have a version of Visual Studio on your windows computer, you can download the latest Express version [here](http://www.visualstudio.com/en-us/products/visual-studio-express-vs.aspx)

In general, I believe that the solutions and projects provided in this project should work on the Express editions, although editing of the solution and/or project
files might be necessary.

### Downloading CUDA
CUDA for Windows can be downloaded [here](https://developer.nvidia.com/cuda-downloads).

Find your OS from the table on that page and download the appropriate version.  Windows users note that there is a notebook and a desktop version of the CUDA
toolset.  Make sure to choose the appropriate one for your machine.  CUDA version 6.5 is supported under Visual Studio 2010, 2012, and 2013.  Download the
32 or 64 bit version of the tools depending on your operating system.  Either version should work with this project.

After downlaoding CUDA, run the installer to install the toolkit and samples.  Visual Studio _must_ be installed before you install the CUDA toolkit
otherwise you will not have access to the CUDA tool chain inside of Visual Studio.

### Downloading this code
This repository can be cloned locally to your workstation by using any valid git tools for Windows.  A combination of cygwin and git (on cygwin) works as well.  Use
the 'Clone in desktop' option to the right.  Or, you can simply download a zip file with the contents of this repository and then unzip it locally.  (Note the 
"Clone in desktop" feature might not work under some browsers.  Specifically there might be problems in Chrome so try a different browser if hitting the
button does nothing).

### Usage
The Visual Studio 2010 solution can be found in the VS2010 folder.  Likewise, the Visual Studio 2012 solution and the 2013 solution can be found in their respective
folders.  The solutions all share the project files (which are located in the individual folders at the same level as the folders that contain the solution), so when
loading up the VS2012 or VS2013 solutions, you will see that the individual projects are tagged as Visual Studio 2010.  This sharing and backwards compatibility
will not affect local development under VS2012 and VS2013.  This setup was done in order to minimize the number of projects that needed to be created and mainteed.

CMake is not required for these solutions.

### Setting up Intellisense
In order for Visual Studio to properly parse the .cu files, you might need to change a setting inside of Visual Studio in order to tell VS to use C++ editing commands
for the .cu files.  To do so:

-  Go to Tools/Options
-  Text Editor/File Extensions
-  In the Extension box, enter ".cu"
-  Choose Microsoft Visual C++ in the Editor box
-  Hit OK

The soltuion and projects have been set so that the wb.h file gets properly parsed by Intellisense when opened.  A change was made to wbCUDA.h in libwb in order
to include the main CUDA files so that CUDA definitions would also be pulled in during Intellisense parsing. 


### Building the Machine Problems

-  Open the solution according to your version of Visual Studio
-  Right click on the machine problem that you are working on and choose "Set as Startup Project"
-  Hit F7 (or your equivalent keyboard shortcut) to build the selected project.  You can alternatively right click on the project and choose "Build" from the context menu

### Writing code
Place all of your code (host and GPU), inside of the .cu file for the specific problem you are working on.  The compiler will "do the right thing" and use the correct
compiler for the correct code.  Doing development in this manner will allow easy copy/paste from your local development to WebGPU for submission.  If you split your
code into a .cu file for GPU code and a .cpp file for host code, you will have to combine this code together when putting this code into WebGPU for testing and submission.

### Testing the Machine Problems
Datasets for the machine problems for this course can be downloaded from the Description page for each individual problem.  Once extracted locally, these problem sets
can then be used with locally built programs for testing before then testing on WebGPU.  Using the instructions for local development (also found on the Description
page for the individual problems), you can run the following:

```
./program -e <expected_output_file> -i <input_file_1>,<input_file_2> -o <output_file> -t <type>
```

All datasets should contain input and expected output files.  See the local development instructions on Description pages for your problem for more information.

Once you are satisfied with your code, you can copy/paste your code from your local machine to WebGPU.  Once on WebGPU, make sure that you again run your code
before submission.


### Notes
The machine problems projects are configured to use virtual and GPU architecture of 2.0 and 2.0 respectively.  To change this behavior:

-  Right click on the project in the project explorer windows and choose "Properties"
-  Choose "Cuda C/C++/Device"
-  Change the "Code Generation" section to the specific values for your architecture

The soltuion and projects will build for Debug and Release configurations for Win32.  There is no x64 platform defined.

These solutions were built for the CUDA 6.5 toolkit.  If you are using a different CUDA Toolkit version, you will likely have to change the vcxproj files to reflect your current version.  Contact me if you need help making these changes.

Exeucables from the programs can be found in the "Debug" or "Release" folder inside the folder of the solution you are using (rather than in the Debug or Release folder within the indivudual projects).

LOCAL DEVELOPMENT IS NOT A REPLACEMENT FOR WEBGPU.  You must still use WebGPU for testing and submitting your code.



