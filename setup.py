from setuptools import setup, find_packages

setup(
    name='simple_lime_explainer',
    version='0.1.0',
    description='A production-quality project demonstrating LIME explanations for a logistic regression model on the Iris dataset',
    author='Your Name',
    author_email='your.email@example.com',
    url='https://github.com/yourusername/simple-lime-explainer',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'scikit-learn>=0.24',
        'lime>=0.2.0.1',
        'numpy>=1.18',
        'pandas>=1.0',
        'matplotlib>=3.0',
        'joblib>=0.14',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
    ],
    python_requires='>=3.6',
)
