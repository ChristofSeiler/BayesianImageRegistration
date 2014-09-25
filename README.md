Bayesian Image Registration
===========================

To compile you will need the latest <a href="http://www.itk.org/ITK/resources/software.html">ITK</a>, <a href="http://eigen.tuxfamily.org/">Eigen</a>, and <a href="http://www.boost.org/">boost</a> libraries.

<pre>
BayesianImageRegistration --help
Allowed options:
--help                 produce help message
--fixed arg            set path to fixed image
--moving arg           set path to moving image
--output-prefix arg    set prefix for all output files
--cx arg               set number of control point in x direction
--cy arg               set number of control point in y direction
--cz arg               set number of control point in z direction
--verbose              save intermediate matrices and images to file
--noOfSteps arg        set number of Gauss-Newton steps
--lam arg              set prior importance: likelihood weight = 1.0, prior 
weight = lam
--Kstd arg             set standard deviation of normal for momenta
--epsilon arg          set leapfrog stepsize
--L arg                set leapfrog integration steps
--T arg                set number of HMC steps
--warpMode             switch to warp mode
--applyMethod arg      set method name for displacement field and warp output
files
--applyStep arg        set step for displacement field and warp output files
--hessianProposal      set to use inverse of the Hessian as the proposal
--pickControlPoint arg set control point for visualization
</pre>
