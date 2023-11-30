# Course Allocation for Professors

## Group member
Utkarsh misra 2022A7PS0325G
Ashit Jain 2022A7PS0606G
Akshat Gosain 2022A7PS0154G

## Introduction

This code is designed to allocate courses to professors based on their preferences and the university's requirements. The allocation process takes into consideration various factors, including professor preferences, categories, and university constraints. Here's an overview of the key aspects:

- Each professor has preferences for specific courses.
- Professors are categorized as FDC, HDC, FDE, or HDE.
- Courses are categorized as FDC, HDC, FDE, or HDE.
- Professors can teach courses in their category or the category below.
- Each professor can teach a maximum of two courses.
- Each course must be assigned to at least one professor.

## Code Overview

The code follows these main steps:

1. Reads the dataset of professors and their preferences.
2. Iterates over professors, assigning courses based on preferences and university requirements.
3. Checks for crashes, such as over-assignment of courses or unassigned courses.

## Installation

To run the code, install the required libraries:

```bash
pip install pandas numpy seaborn matplotlib networkx
