import pandas as pd
import ast
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import pandas as pd
import ast
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm



# Function to plot the occurrence distribution for 'TOPIC_LABEL'

def plot_topic_label_distribution(filtered_df):
    # Expand the 'TOPIC_LABEL' list into separate rows
    if type(filtered_df['TOPIC_LABEL'].iloc[0]) != list:
        filtered_df['TOPIC_LABEL'] = filtered_df['TOPIC_LABEL'].apply(ast.literal_eval)

    expanded_df = filtered_df.explode('TOPIC_LABEL')

    # Group by 'TOPIC_LABEL' and count occurrences
    topic_counts = expanded_df.groupby('TOPIC_LABEL').size().reset_index(name='count')

    print(topic_counts)

    # Plot bar chart
    plt.bar(topic_counts['TOPIC_LABEL'], topic_counts['count'])
    plt.xlabel('TOPIC_LABEL')
    plt.ylabel('Occurrences')
    plt.title('Occurrence distribution by TOPIC_LABEL')
    plt.show()



# Function to plot the length of text in the 'RESULTS' column by index

def plot_result_length(filtered_df):
    # Calculate the length of the text in the 'RESULTS' column
    lengths = filtered_df['RESULTS'].astype(str).apply(len)
    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(lengths, marker='o')
    plt.xlabel('Index')
    plt.ylabel('Text length in RESULTS')
    plt.title('Text length in the RESULTS column by index')
    plt.show()



# Function to plot and analyze Gaussian distribution of text lengths in 'RESULTS' column

def plot_gaussian_length_distribution(filtered_df):
    # Calculate the length of text in the 'RESULTS' column
    lengths = filtered_df['RESULTS'].astype(str).apply(len)

    # Calculate statistics
    min_length = lengths.min()
    max_length = lengths.max()
    mean_length = lengths.mean()
    std_length = lengths.std()

    print(f"STATISTICS FOR 'RESULTS' LENGTH")
    print(f"Minimum: {min_length}")
    print(f"Maximum: {max_length}")
    print(f"Mean: {mean_length:.2f}")
    print(f"Standard deviation: {std_length:.2f}")
    print("-" * 40)

    # Normalized histogram (density=True)
    plt.figure(figsize=(10, 5))
    n, bins, patches = plt.hist(lengths, bins=20, density=True, alpha=0.6, color='g', label='Histogram')

    # Fit normal distribution parameters
    mu, std = norm.fit(lengths)

    # Overlay Gaussian curve
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, 'k', linewidth=2, label=f'Gaussian\n($\\mu$={mu:.1f}, $\\sigma$={std:.1f})')

    plt.xlabel('Text length in RESULTS')
    plt.ylabel('Density')
    plt.title('Text Length Distribution in RESULTS\nwith Gaussian Curve')
    plt.legend()
    plt.show()



# Import necessary libraries
import matplotlib.pyplot as plt
import ast
import numpy as np
from scipy.stats import norm

# Function to plot both topic label distribution and Gaussian length distribution in the same figure
def plot_topic_label_and_gaussian_length_distribution(filtered_df):
    # Expand the 'TOPIC_LABEL' list into separate rows if not already a list of lists
    if type(filtered_df['TOPIC_LABEL'].iloc[0]) != list:
        filtered_df['TOPIC_LABEL'] = filtered_df['TOPIC_LABEL'].apply(ast.literal_eval)

    expanded_df = filtered_df.explode('TOPIC_LABEL')

    # Group by 'TOPIC_LABEL' and count occurrences
    topic_counts = expanded_df.groupby('TOPIC_LABEL').size().reset_index(name='count')

    # Calculate the length of text in the 'RESULTS' column
    lengths = filtered_df['RESULTS'].astype(str).apply(len)

    # Prepare for Gaussian fit
    mu, std = norm.fit(lengths)

    # Start plotting
    fig, axes = plt.subplots(2, 1, figsize=(12, 12), constrained_layout=True)

    # First subplot: Bar plot for topic label distribution
    axes[0].bar(topic_counts['TOPIC_LABEL'].astype(str), topic_counts['count'], color='skyblue')
    axes[0].set_xlabel('TOPIC_LABEL')
    axes[0].set_ylabel('Occurrences')
    axes[0].set_title('Occurrence distribution by TOPIC_LABEL')
    axes[0].tick_params(axis='x', rotation=45)

    # Second subplot: Histogram and Gaussian fit for text length
    n, bins, patches = axes[1].hist(lengths, bins=20, density=True, alpha=0.6, color='g', label='Histogram')
    xmin, xmax = axes[1].get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    axes[1].plot(x, p, 'k', linewidth=2, label=f'Gaussian\n($\\mu$={mu:.1f}, $\\sigma$={std:.1f})')
    axes[1].set_xlabel('Text length in RESULTS')
    axes[1].set_ylabel('Density')
    axes[1].set_title('Text Length Distribution in RESULTS with Gaussian Curve')
    axes[1].legend()

    plt.show()