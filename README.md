# Advance-Facial-Recogntion-Using-HBDSCAN-Lab-4-AI-
```markdown
# Advanced Facial Recognition System with HDBSCAN 🧠📸  

## 🚀 Project Overview  
This project implements an **Advanced Facial Recognition System** using **Principal Component Analysis (PCA)** for dimensionality reduction and **HDBSCAN** for unsupervised clustering. It efficiently groups facial data without supervision, making it suitable for scenarios where labeled datasets are unavailable.  

---

## 🛠️ Features  
- **Dimensionality Reduction**: Compresses high-dimensional facial data using PCA.  
- **Unsupervised Clustering**: Identifies clusters of faces based on similarity using HDBSCAN.  
- **Visualizations**: Generates 2D/3D plots of clusters for better interpretability.  
- **Scalable**: Designed to handle large datasets effectively.  

---

## 📂 File and Folder Structure  
```plaintext  
|-- lwf/                # Contains input facial datasets    
|-- step_1 (loading_and_exploring).py          #first step of the lab and first file you should run   
|-- step_2 (data_preprocessing).py             #second step of the lab and second file that you should run  
|-- analysis.py             #after running step 2, you can run this file so as to interprete the results of the file that gets generated i.e the preprocessed_features.npy file 
|-- main.py            #script that contains PCA and HBDSCAN clustering and should be run after step 1 and 2 
|-- README.md            # Project documentation  
```  

---

## 📋 Requirements  
- Python 3.8+ 🐍  
- Libraries:  
  - `numpy`  
  - `pandas`  
  - `scikit-learn`  
  - `hdbscan`  
  - `matplotlib`  
  - `seaborn`  

Install dependencies with:  
```bash  
pip install -r requirements.txt  
```  

---

## 🧑‍💻 Usage  
1. **Prepare the Dataset**:  
   Place your dataset in the `lfw/` folder.  

2. **Preprocess the Dataset**:  
   Perform data preprocessing by executing:  
   ```bash  
   python step_2 (data_preprocessing).py  
   ```  
3. **Run PCA**:  
   Reduce the dataset's dimensionality by executing:  
   ```bash  
   python main.py   
   ```  

4. **Perform Clustering**:  
   Apply HDBSCAN to identify clusters:  
   ```bash  
   python main.py  
   ```  
---

## 📊 Outputs  
- **Cluster Assignments**: Labels indicating each face's group.  
- **Visualization**: 2D and 3D scatter plots of PCA-reduced data with HDBSCAN clusters.  

---

## 🤔 Why HDBSCAN?  
HDBSCAN excels in clustering non-spherical data, handling noise, and working with varying densities, making it ideal for complex datasets like facial embeddings.  

---

## 👏 Contributions  
Feel free to fork, star ⭐, and contribute to this project. Submit a pull request or open an issue for discussions.  

---

Happy clustering! 🎉  
```

Let me know if you'd like to tweak or expand on any section!
