# iris_mnist_pca_assignment
# Machine Learning Assignment: K-Means Clustering and PCA Analysis

**Student:** Abishek 
**Date:** September 21, 2025  
**Course:** Intro to machine language  

---

## Executive Summary

This assignment explores two fundamental machine learning techniques: the elbow method for optimal cluster determination in K-means clustering and Principal Component Analysis (PCA) for dimensionality reduction in logistic regression. The analysis reveals key insights about when these techniques are most beneficial and their computational trade-offs.

---

## Part 1: Elbow Method for K-Means Clustering

### Objective
Determine the optimal number of clusters (k) for the Iris dataset using the elbow method.

### Dataset
- **Source:** Scikit-learn Iris dataset
- **Features:** 4 (sepal length/width, petal length/width)
- **Samples:** 150
- **Classes:** 3 (setosa, versicolor, virginica)

### Methodology
1. Applied K-means clustering for k values ranging from 1 to 10
2. Calculated Within-Cluster Sum of Squares (WCSS) for each k value
3. Plotted WCSS vs. number of clusters to identify the "elbow"
4. Used consistent random state (42) for reproducibility

### Results

```
k=1: inertia=681.37
k=2: inertia=152.35
k=3: inertia=78.85
k=4: inertia=57.23
k=5: inertia=46.45
k=6: inertia=39.04
k=7: inertia=34.05
k=8: inertia=29.91
k=9: inertia=27.02
k=10: inertia=23.87
```

### Analysis
- **Clear Elbow at k=3:** The most significant drop in WCSS occurs between k=2 and k=3
- **Diminishing Returns:** Beyond k=3, WCSS reduction becomes gradual
- **Perfect Alignment:** The optimal k=3 matches the known 3 species in the Iris dataset
- **Validation:** This demonstrates the elbow method's effectiveness for natural cluster identification

### Interpretation
The elbow method successfully identified the optimal number of clusters, proving its utility as an unsupervised technique for cluster validation. The sharp bend at k=3 indicates that adding more clusters beyond this point provides minimal improvement in data explanation.

---

## Part 2: PCA Impact on Logistic Regression Performance

### Objective
Compare logistic regression performance and training time with and without PCA dimensionality reduction on the MNIST dataset.

### Dataset
- **Source:** OpenML MNIST handwritten digits
- **Original Features:** 784 (28×28 pixel values)
- **Samples:** 70,000 total (56,000 training, 14,000 testing)
- **Classes:** 10 (digits 0-9)

### PCA Configuration
- **Variance Retention:** 95% of total variance
- **Components Retained:** 330 out of 784 features
- **Dimensionality Reduction:** 58% reduction (454 fewer features)

### Experimental Setup
1. **Data Preprocessing:** StandardScaler normalization
2. **Train-Test Split:** 80/20 stratified split
3. **PCA Settings:** 95% variance retention, random_state=42
4. **Logistic Regression:** LBFGS solver, max_iter=200, single-threaded

### Results Summary

| Metric | Without PCA | With PCA (95%) | Difference | Impact |
|--------|-------------|----------------|------------|---------|
| **Accuracy** | 0.9197 | 0.9197 | 0.0000 | No change |
| **Training Time** | 18.816s | 20.890s | +2.074s | 11% slower |
| **PCA Time** | 0.000s | 5.666s | +5.666s | Preprocessing overhead |
| **Total Time** | 18.816s | 26.556s | +7.740s | 41% slower overall |
| **Features** | 784 | 330 | -454 | 58% reduction |
| **Memory Usage** | 100% | ~42% | -58% | Significant savings |

### Detailed Analysis

#### Accuracy Performance
- **Identical Accuracy:** Both approaches achieved 91.97% accuracy
- **No Information Loss:** 95% variance retention preserved predictive power
- **Robust Classification:** PCA maintained digit classification quality

#### Training Time Analysis
**Unexpected Result:** PCA increased total processing time by 41%

**Contributing Factors:**
1. **PCA Computation Overhead:** 5.666 seconds for dimensionality reduction
2. **High-Dimensional Input:** 784 features require substantial PCA computation
3. **Remaining Complexity:** 330 components still computationally significant
4. **Dataset Size:** 56,000 training samples amplify PCA costs

#### Memory and Storage Benefits
- **58% Feature Reduction:** Significant storage savings
- **Model Complexity:** Simpler coefficient matrix (330 vs. 784 parameters)
- **Deployment Efficiency:** Smaller models for production environments

### When PCA Helps vs. Hurts

#### PCA Advantages:
- **Memory-Constrained Environments:** Significant storage reduction
- **Very High Dimensionality:** >1000 features with sparse information
- **Noise Reduction:** Eliminates low-variance noise components
- **Visualization:** Enables 2D/3D data representation
- **Multicollinearity:** Removes correlated feature dependencies

#### PCA Disadvantages:
- **Computational Overhead:** Expensive for moderate-sized datasets
- **Interpretability Loss:** Principal components lack direct meaning
- **Information Loss:** Some predictive signal may be discarded
- **Training Time:** Can increase total processing time

### Real-World Implications

#### Production Considerations
1. **One-Time Cost:** PCA computation is performed once, reused for inference
2. **Inference Speed:** Reduced feature space accelerates prediction time
3. **Model Storage:** 58% smaller models for deployment
4. **Bandwidth:** Lower data transfer costs for distributed systems

#### Recommendation Framework
**Use PCA When:**
- Memory/storage is severely constrained
- Feature count > 1000 with redundant information
- Deploying models to resource-limited devices
- Visualization of high-dimensional data is needed

**Avoid PCA When:**
- Training time is more critical than storage
- Feature interpretability is essential
- Dataset dimensionality is already manageable
- Computational resources are abundant

---

## Conclusion

This analysis demonstrates that machine learning technique selection requires careful consideration of specific constraints and objectives:

1. **Elbow Method Success:** Effectively identified optimal cluster count, validating its utility for unsupervised learning tasks

2. **PCA Trade-offs:** While achieving 58% dimensionality reduction with no accuracy loss, PCA increased training time by 41% due to computational overhead

3. **Context Matters:** The effectiveness of dimensionality reduction techniques depends heavily on dataset characteristics, computational resources, and deployment requirements

4. **Real-World Insight:** Academic examples often show PCA speeding up training, but this analysis reveals the nuanced reality where preprocessing costs can dominate

### Key Takeaway
Both techniques are powerful tools in the machine learning toolkit, but their application requires understanding the specific trade-offs between accuracy, computational efficiency, storage requirements, and interpretability for each unique use case.

---

## Technical Implementation Notes

### Environment Setup
- **Python Version:** 3.11
- **Key Libraries:** scikit-learn 1.7.2, threadpoolctl 3.6.0
- **Threading:** Single-threaded execution to avoid system conflicts
- **Reproducibility:** Fixed random states for consistent results

### Code Optimizations
- Environment variables set to prevent threading conflicts
- Explicit algorithm parameters for stability
- Comprehensive error handling for robust execution
- Performance timing for accurate measurements

---

