#PS Entire thing scrapped and redone in new repo.

List of all [commands](Broomburg/commands.txt)
- Uses a tkinter interface (ew).  
- Raw: able to pull live price and fundamental data via yahooquerry and news via RSS feeds.  
- Processed: relative valuation (factors), auto populate data necessary for dcf onto xlsx file, basic vol calculations, bad attempts at MATCH interface for screening and shell interface. Prob missed some stuff.  

# EDA
#PS Very, very small sample size to speed up run time of tests.
Reason is this is just a proof of concept, not meant to be statistically significant. 

HML3
![HML3](Broomburg/data/output/eda/num_HML_3.png)

Sector
![Sector](Broomburg/data/output/eda/cat_sectorKey.png)


# OLS Inference
Linearity, Homoscedasticity, Independence and Normality of Errors
![Linearity, Homoscedasticity, Independence and Normality of Errors](Broomburg/data/output/eda/eda_reg_residuals.png)

Leverage and Multicollinearity
![Leverage and Multicollinearity](Broomburg/data/output/eda/eda_reg_diagnostics.png)


# XGB
Learning Curve
![Learning_Curve](Broomburg/data/output/eda/XGB1_learning_curve_rmse.png)

Dependency
![Dependency](Broomburg/data/output/eda/XGB1_dependency_diag.png)

Explanation
![Explanation](Broomburg/data/output/eda/XGB1_explanation_diag.png)

# Random Forest
SHAP
![SHAP](Broomburg/data/output/eda/RF1_shap_summary.png)


# PCA
Scree + CEV
![PCA](Broomburg/data/output/eda/PCA_Variance.png)

Loadings
![PCA](Broomburg/data/output/eda/PCA_Loadings.png)

# Corr
Returns
![Correlation](Broomburg/data/output/eda/corr_Returns_pearson.png)



