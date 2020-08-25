Map of this repository:

	===========================
	Scripts that make heatmaps
	===========================
- heatmapCsvMaker
- heatmapFromCsv
- histoFromCsv

To make a heatmap, first run heatmapCsvMaker. This script does the number crunching and creates a csv with all the numbers for a heatmap. For a sense of the runtime, for these parameters: 
- nMax=10
- dx=0.05
- bounds: [-20,20]
- numRegressions=100
the runtime would be ~20 minutes on Erwin.

Then, run heatmapFromCsv. This script reads the csv created by heatmapCsvMaker and produces the actual heatmap.
Alternatively, run histoFromCsv. This script is similar to heatmapFromCsv, but creates a histogram of intercepts.



	================================
	Scripts for visualizing a sample
	================================
- lnSigmaLines:      linear ln(sigma)-ln(N) plot
- histo-overlaps:    histogram of <psi|zeta><zeta|phi> values,
                     the average of which should be <psi|phi>



	==================================================
	Scripts for visualizing potentials and eigenstates
	==================================================
- energyLevels:         energy level diagram
- gifEnergyLevels:      animated energy level diagram
- wavefunctionDisplay:  plot 1 or 2 energy eigenfunctions
                        for a given potential



	===============
	Support scripts
	===============
- sho:        potentials and eigensolvers
- myStats:    my homemade stats package

sho contains 4 types of functions:
1. functions that define some potential U
2. a function named potentialEigenstates, which numerically
   solves for the eigenstates of a given potential U
3. functions that just call another function to define
   a potential, then feed that potential to potentialEigenstates
   and return the eigenstates
4. a function named shoEigenket that returns the _analytic_ 
   solutions to the simple harmonic oscillator

myStats contains very basic stats (mean, stdev) and a linear regression function. There's nothing here that you couldn't find in some standard python package.



	===========
	   Misc.
	===========
eigenConvergence can be used to evaluate at what mesh size energy values converge. See 8/14 work log for some explanation of the convergence criterion I implemented.