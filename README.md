# [Visual Product Matcher](https://visual-product-matcher-1hkn.onrender.com/)

> A web-based visual product search application that finds similar fashion products using image embeddings and a hybrid machine learning model based on ResNet50 and OpenAI CLIP.

[![Live Demo](https://img.shields.io/badge/demo-live-brightgreen)](https://visual-product-matcher-1hkn.onrender.com/)
[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

![Product Matcher Demo](./demo.GIF)

##  Author
**Anik Halder**  
(24MCA0251)  
🎓 VIT Vellore

## Table of Contents  
- [Features](#features)  
- [Demo](#demo)
- [Project Report](#project-report)    
- [Tech Stack](#tech-stack)  
- [Architecture](#architecture)  
- [Performance](#performance)  
- [Installation](#installation)  
- [License](#license)  
  
## Features

- Upload product images to find visually similar items
- Hybrid similarity matching using ResNet50, CLIP, and text embeddings
- Filter products by category and subcategory
- Responsive mobile-friendly design
- Real-time product search with 90%+ accuracy

## Demo

**Live Application:** [Products Visual Match](https://visual-product-matcher-1hkn.onrender.com/)

## Project Report

- [Project Report (PDF)](./Project%20Report.pdf)  

### How It Works

1. Browse all products in the catalogue
2. Upload a Product Image
3. View similar products ranked by similarity score

## Tech Stack

### Frontend
- **Flask** - Python web framework
- **Bootstrap 5** - Responsive UI
- **Vanilla JavaScript** - Dynamic content loading

### Backend
- **FastAPI** - High-performance API
- **MongoDB Atlas** - Cloud database
- **TensorFlow/Keras** - ResNet50 model
- **OpenAI CLIP** - Visual + text embeddings

### Deployment
- **Web App:** Render
- **API:** HuggingFace Spaces
- **Database:** MongoDB Atlas

## Architecture

The system uses a **3-stage similarity ranking algorithm**:

### Stage 1: Visual Ranking
- Combines ResNet50 (60%) + CLIP image embeddings (40%)
- Identifies top 100 visually similar products

### Stage 2: Semantic Filtering
- Filters by text similarity threshold (0.25)
- Prevents gender/category confusion

### Stage 3: Hybrid Re-ranking
- Final score: Visual (60%) + Semantic (40%)
- Returns top K most similar products

**Benefits:**
- ✅ High visual accuracy for colors, patterns, textures
- ✅ Correct gender/category matching
- ✅ Fast response times (~1-2 seconds)

## Performance

| Metric | Score |
|--------|-------|
| **Visual Similarity** | 90%+ match for colors, patterns |
| **Semantic Accuracy** | 95%+ correct gender/category |
| **Response Time** | 1-2 seconds per query |
| **Dataset Size** | 225 products |
| **Embedding Dimensions** | 2048 (ResNet), 512 (CLIP) |

## Installation

### Prerequisites

- Python 3.9 or higher
- pip package manager
- Git

### Quick Start

#### ✔ Clone repository

> git clone https://github.com/Anikrage/Visual_Product_Matcher.git  
> cd Visual_Product_Matcher  

#### ✔ Install dependencies

>cd webapp
> pip install -r requirements.txt  

#### ✔ Database Access

> Database access is configured for read-only test access.  
> Email @ (anik.halder2024@vitstudent.ac.in) for Database Access and create .env

#### ✔ Run application

> cd webapp  
> python app.py  


## License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

