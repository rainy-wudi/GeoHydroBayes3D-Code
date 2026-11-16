# Source provenance

The analysis scripts implement the corrected random-field and final-groundwater-boundary workflow used in the manuscript. They retain the production network architecture, priors, covariance estimates, likelihoods and numerical summary algorithms. Input paths are relative to the repository, and optional environment variables support shorter validation runs.

The MATLAB code includes the production random-field generator, mesh reader and model template. Its default SMOKE mode prepares a limited finite-element test before all 1000 inputs are generated. Comments and diagnostic messages are provided in English.

The included engineering data, borehole observations, trained models and numerical reference results originate from the study and are released with the authors' permission. See VALIDATION.md for the checks performed and their limitations.
