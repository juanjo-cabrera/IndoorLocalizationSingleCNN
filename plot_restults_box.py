import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

ruta = '/home/arvc/Juanjo/develop/Extension Orlando/'
# models_names = ['AlexNet', 'resnet_152', 'convnext', 'resnext', 'efficientnet', 'mobilenet']
models_names = ['convnext']
training_sequences_names = ['noDA', 'DA1', 'DA2', 'DA3', 'DA4', 'DA5', 'DA6']
illuminations = ['cloudy', 'night', 'sunny']

font = 15
base_results = '/home/arvc/Juanjo/develop/Extension Orlando/'
all_data = []

# Read data from CSV files
for model_base_name in models_names:
    for dataset_name in training_sequences_names:
        global_data = []
        for illumination in illuminations:
            # Construct the file path
            results = base_results + illumination + '_' + model_base_name + '_' + dataset_name + '.csv'

            # Check if the file exists
            if os.path.exists(results):
                # Read the CSV file
                df = pd.read_csv(results)

                # Ensure 'Errors' column exists
                if 'Errors' in df.columns:
                    # Unpack the 'Errors' lists into individual rows
                    for error in df['Errors']:
                        all_data.append({
                            'Errors': float(error.strip('[]')),
                            'Model': model_base_name,
                            'Training Sequence': dataset_name,
                            'Illumination': illumination.capitalize()
                        })
                        global_data.append({
                            'Errors': float(error.strip('[]')),
                            'Model': model_base_name,
                            'Training Sequence': dataset_name,
                            'Illumination': 'Global'
                        })
                else:
                    print(f"'Errors' column not found in {results}")


            else:
                print(f"File not found: {results}")
        all_data = all_data + global_data


illuminations = [s.capitalize() for s in illuminations]
illuminations.append('Global')

# Create a DataFrame from the unpacked data
if all_data:
    print(all_data)
    all_data_df = pd.DataFrame(all_data)

    # Print the first few rows of the dataframe to check if data is read correctly
    print(all_data_df.head())

    # Define custom color palette
    custom_palette = sns.color_palette("Spectral", len(illuminations) + 1)

    # Plot the data using Seaborn
    plt.figure(figsize=(15, 10))
    ax = sns.boxplot(x='Training Sequence', y='Errors', hue='Illumination', data=all_data_df, showfliers=False,
                     palette=custom_palette, hue_order=['Cloudy', 'Night', 'Sunny', 'Global'])
    ax.set_xlabel('Backbone Models', fontsize=font)
    ax.set_ylabel('Hierarchical Localization Error (m)', fontsize=font)
    ax.set_title('Hierarchical Localization Error for Different Backbone Models', fontsize=font)
    ax.legend(fontsize=font)
    # Draw horizontal grid lines
    # plt.grid(axis='y', which='minor')
    # Draw main grid lines and subgrid lines
    plt.grid(axis='y', which='major', linestyle='-', linewidth='0.5', color='gray')
    plt.grid(axis='y', which='minor', linestyle=':', linewidth='0.5', color='gray')
    ax.minorticks_on()

    # Calculate and annotate the mean values
    mean_values = all_data_df.groupby(['Training Sequence', 'Illumination'])['Errors'].mean().reset_index()
    for i in range(len(mean_values)):
        seq = mean_values.loc[i, 'Training Sequence']
        illum = mean_values.loc[i, 'Illumination']
        mean_val = mean_values.loc[i, 'Errors']

        x = training_sequences_names.index(seq)

        print(x)
        factor = 0.2
        hue_width = len(illuminations) * factor
        hue_offset = -hue_width / 2 + illuminations.index(illum) * factor  + factor  / 2
        # hue_offset = {'cloudy': -0.27, 'night': 0, 'sunny': 0.27, 'global': 0.40}  # Adjust offset according to hue categories
        ax.text(x + hue_offset, - 0.09 , f'{mean_val:.2f}',
                ha='center', va='bottom', color='black', fontsize=font*0.9)
        ax.scatter(x + hue_offset, mean_val, color='black', s=100, zorder=3)  # zorder to draw on top of the boxes

    plt.xlim(- hue_width*2/3, len(training_sequences_names) - 1 + hue_width*2/3)  # Ajusta los límites del eje x
    plt.ylim(-0.115, 2.5)  # Ajusta los límites del eje y
    plt.xticks(fontsize=font)
    plt.yticks(fontsize=font)
    plt.subplots_adjust(left=0.04, right=0.995, bottom=0.06, top=0.95)
    plt.show()
else:
    print("No data available to plot.")

# Create a DataFrame from the unpacked data
if all_data:
    all_data_df = pd.DataFrame(all_data)

    # Print the first few rows of the dataframe to check if data is read correctly
    print(all_data_df.head())
    # Define custom color palette
    custom_palette = sns.color_palette("Spectral", len(illuminations))

    # Plot the data using Seaborn
    plt.figure(figsize=(15, 10))
    ax = sns.boxplot(x='Training Sequence', y='Errors', hue='Illumination', data=all_data_df,
                     palette=custom_palette)
    plt.title('Hierarchical Localization Error for Different Backbone Models')
    plt.xlabel('Training Dataset')
    plt.ylabel('Hierarchical Localization Error (m)')
    plt.legend()
    # Draw horizontal grid lines
    # plt.grid(axis='y', which='minor')
    # Draw main grid lines and subgrid lines
    plt.grid(axis='y', which='major', linestyle='-', linewidth='0.5', color='gray')
    plt.grid(axis='y', which='minor', linestyle=':', linewidth='0.5', color='gray')
    ax.minorticks_on()

    # Calculate and annotate the mean values
    mean_values = all_data_df.groupby(['Training Sequence', 'Illumination'])['Errors'].mean().reset_index()
    for i in range(len(mean_values)):
        seq = mean_values.loc[i, 'Training Sequence']
        illum = mean_values.loc[i, 'Illumination']
        mean_val = mean_values.loc[i, 'Errors']

        x = training_sequences_names.index(seq)

        print(x)
        hue_offset = {'cloudy': -0.25, 'night': 0, 'sunny': 0.25}  # Adjust offset according to hue categories
        ax.text(x + hue_offset[illum], - 0.6 , f'{mean_val:.2f}',
                ha='center', va='bottom', color='black', fontsize=10)

    plt.show()
else:
    print("No data available to plot.")