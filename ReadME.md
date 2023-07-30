Heart failure is a disease that affects 5.5 million Americans and occurs when blood is unable to reach other organs. When this occurs, an infarct (or area with dead heart tissue) is developed. The larger the infarct the more severe heart failure is. To assess the state of a patient and help with symptoms, many people who have heart failure are placed on temporary devices (Impella or ECMO). Impella devices work from an Archimedes-like screw pump which pushes blood out of the left ventricle into the aorta. Extracorporeal membrane oxygenation (ECMO) is a system that gives biventricular support to patients. In other words, ECMO can artificially support the heart and lungs. However, oftentimes these devices can lead to complications and many people die. 
Thus, I developed four algorithms that used hemodynamic features to predict infarct size which will further help evaluate the severity of heart failure in a patient. Specifically, K nearest neighbors, XGBoost, and regression models (linear and polynomial) were used to predict infarct size. All four models achieved low error rates (<5 %) as well as high R2 values (>90%) in terms of predicting infarct size. Through interpretability and analyzing data distributions, it was found that the impella group had better cardiovascular health compared to the ECMO group. Thus, this project not only allows us to evaluate the severity of heart failure in a patient, but it also improves our understanding of the hemodynamic differences between ECMO and impella.