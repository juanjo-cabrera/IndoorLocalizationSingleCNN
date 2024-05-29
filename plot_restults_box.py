import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

ruta = '/home/arvc/Juanjo/develop/Extension Orlando/'
# models_names = ['AlexNet', 'resnet_152', 'convnext', 'resnext', 'efficientnet', 'mobilenet']
models_names = ['convnext']
training_sequences_names = ['noDA', 'DA1', 'DA2', 'DA3', 'DA4', 'DA5', 'DA6']
plot_names = ['Baseline', 'DA 1', 'DA 2', 'DA 3', 'DA 4', 'DA 5', 'DA 6']
illuminations = ['cloudy', 'night', 'sunny']

font = 25
base_results = '/home/arvc/Juanjo/develop/Extension Orlando/'
all_data = []

# Read data from CSV files
for model_base_name in models_names:
    for dataset_name, plot_name in zip(training_sequences_names, plot_names):
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
                            'Training Sequence': plot_name,
                            'Illumination': illumination.capitalize()
                        })
                        global_data.append({
                            'Errors': float(error.strip('[]')),
                            'Model': model_base_name,
                            'Training Sequence': plot_name,
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
    ax.set_title('Hierarchical Localization Error for Different Training Datasets', fontsize=font)
    ax.legend(fontsize=font)
    # Draw horizontal grid lines
    # plt.grid(axis='y', which='minor')
    # Draw main grid lines and subgrid lines
    plt.grid(axis='y', which='major', linestyle=':', linewidth='0.4', color='gray')
    plt.grid(axis='y', which='minor', linestyle=':', linewidth='0.2', color='gray')
    ax.minorticks_on()

    # Calculate and annotate the mean values
    mean_values = all_data_df.groupby(['Training Sequence', 'Illumination'])['Errors'].mean().reset_index()
    for i in range(len(mean_values)):
        seq = mean_values.loc[i, 'Training Sequence']
        illum = mean_values.loc[i, 'Illumination']
        mean_val = mean_values.loc[i, 'Errors']

        x = plot_names.index(seq)

        print(x)
        factor = 0.2
        hue_width = len(illuminations) * factor
        hue_offset = -hue_width / 2 + illuminations.index(illum) * factor + factor  / 2
        # hue_offset = {'cloudy': -0.27, 'night': 0, 'sunny': 0.27, 'global': 0.40}  # Adjust offset according to hue categories
        vertical_offset = {'Cloudy': 0.075, 'Night': 0.12, 'Sunny': 0.09, 'Global': 0.09}
        # vertical_offset = {'Cloudy': 0.075, 'Night': 0.12, 'Sunny': 0.09, 'Global': 0.09}
        # ax.text(x + hue_offset, mean_val + vertical_offset[illum] , f'{mean_val:.2f}',
        #         ha='center', va='bottom', color='black', fontsize=font)
        ax.text(x + hue_offset, - 0.18, f'{mean_val:.2f}',
                ha='center', va='bottom', color='black', fontsize=font * 0.9, rotation = 30)
        ax.scatter(x + hue_offset, mean_val, color='black', s=100, zorder=3)  # zorder to draw on top of the boxes

    plt.xlim(- hue_width*2/3, len(training_sequences_names) - 1 + hue_width*2/3)  # Ajusta los límites del eje x
    plt.ylim(-0.18, 2.42)  # Ajusta los límites del eje y
    plt.xticks(fontsize=font)
    plt.yticks(fontsize=font)
    plt.subplots_adjust(left=0.06, right=0.995, bottom=0.082, top=0.95)
    plt.show()
else:
    print("No data available to plot.")

# # Create a DataFrame from the unpacked data
# if all_data:
#     all_data_df = pd.DataFrame(all_data)
#
#     # Print the first few rows of the dataframe to check if data is read correctly
#     print(all_data_df.head())
#     # Define custom color palette
#     custom_palette = sns.color_palette("Spectral", len(illuminations))
#
#     # Plot the data using Seaborn
#     plt.figure(figsize=(15, 10))
#     ax = sns.boxplot(x='Training Sequence', y='Errors', hue='Illumination', data=all_data_df,
#                      palette=custom_palette)
#     plt.title('Hierarchical Localization Error for Different Backbone Models')
#     plt.xlabel('Training Dataset')
#     plt.ylabel('Hierarchical Localization Error (m)')
#     plt.legend()
#     # Draw horizontal grid lines
#     # plt.grid(axis='y', which='minor')
#     # Draw main grid lines and subgrid lines
#     plt.grid(axis='y', which='major', linestyle='-', linewidth='0.5', color='gray')
#     plt.grid(axis='y', which='minor', linestyle=':', linewidth='0.5', color='gray')
#     ax.minorticks_on()
#
#     # Calculate and annotate the mean values
#     mean_values = all_data_df.groupby(['Training Sequence', 'Illumination'])['Errors'].mean().reset_index()
#     for i in range(len(mean_values)):
#         seq = mean_values.loc[i, 'Training Sequence']
#         illum = mean_values.loc[i, 'Illumination']
#         mean_val = mean_values.loc[i, 'Errors']
#
#         x = training_sequences_names.index(seq)
#
#         print(x)
#         hue_offset = {'cloudy': -0.25, 'night': 0, 'sunny': 0.25}  # Adjust offset according to hue categories
#         ax.text(x + hue_offset[illum], - 0.6 , f'{mean_val:.2f}',
#                 ha='center', va='bottom', color='black', fontsize=10)
#
#     plt.show()
# else:
#     print("No data available to plot.")

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

ruta = '/home/arvc/Juanjo/develop/Extension Orlando/'
models_names = ['AlexNet', 'resnet_152', 'resnext', 'mobilenet', 'efficientnet', 'convnext']
plot_names = ['AlexNet', 'ResNet-152', 'ResNeXt-101', 'MobileNetV3', 'EfficientNetV2', 'ConvNeXt Large']
# models_names = ['convnext']
training_sequences_names = ['noDA']
illuminations = ['cloudy', 'night', 'sunny']

font = 25
base_results = '/home/arvc/Juanjo/develop/Extension Orlando/'
all_data = []

# Read data from CSV files
for model_base_name, plot_name in zip(models_names, plot_names):
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
                            'Model': plot_name,
                            'Training Sequence': dataset_name,
                            'Illumination': illumination.capitalize()
                        })
                        global_data.append({
                            'Errors': float(error.strip('[]')),
                            'Model': plot_name,
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
    ax = sns.boxplot(x='Model', y='Errors', hue='Illumination', data=all_data_df, showfliers=False,
                     palette=custom_palette, hue_order=['Cloudy', 'Night', 'Sunny', 'Global'])
    ax.set_xlabel('Backbone Models', fontsize=font)
    ax.set_ylabel('Hierarchical Localization Error (m)', fontsize=font)
    ax.set_title('Hierarchical Localization Error for Different Backbone Models', fontsize=font)
    ax.legend(fontsize=font)
    # Draw horizontal grid lines
    # plt.grid(axis='y', which='minor')
    # Draw main grid lines and subgrid lines
    plt.grid(axis='y', which='major', linestyle=':', linewidth='0.2', color='gray')
    plt.grid(axis='y', which='minor', linestyle=':', linewidth='0.1', color='gray')
    ax.minorticks_on()

    # Calculate and annotate the mean values
    mean_values = all_data_df.groupby(['Model', 'Illumination'])['Errors'].mean().reset_index()
    for i in range(len(mean_values)):
        seq = mean_values.loc[i, 'Model']
        illum = mean_values.loc[i, 'Illumination']
        mean_val = mean_values.loc[i, 'Errors']

        x = plot_names.index(seq)

        print(x)
        factor = 0.2
        hue_width = len(illuminations) * factor
        hue_offset = -hue_width / 2 + illuminations.index(illum) * factor  + factor  / 2
        # hue_offset = {'cloudy': -0.27, 'night': 0, 'sunny': 0.27, 'global': 0.40}  # Adjust offset according to hue categories
        ax.text(x + hue_offset, - 1.2 , f'{mean_val:.2f}',
                ha='center', va='bottom', color='black', fontsize=font*0.95, rotation = 30)
        ax.scatter(x + hue_offset, mean_val, color='black', s=100, zorder=3)  # zorder to draw on top of the boxes

    plt.xlim(- hue_width*2/3, len(models_names) - 1 + hue_width*2/3)  # Ajusta los límites del eje x
    plt.ylim(-1.2, 14.8)  # Ajusta los límites del eje y
    plt.xticks(fontsize=font)
    plt.yticks(fontsize=font)
    plt.subplots_adjust(left=0.055, right=0.995, bottom=0.08, top=0.95)
    plt.show()
else:
    print("No data available to plot.")