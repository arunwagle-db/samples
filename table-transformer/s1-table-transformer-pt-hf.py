# Databricks notebook source
# !pip install transformers==4.25.1
# !ls -l /databricks/jars
# !pip install timm

# COMMAND ----------

# MAGIC %md
# MAGIC All Imports go here

# COMMAND ----------

from PIL import Image
from transformers import DetrFeatureExtractor,  TableTransformerForObjectDetection, TableTransformerForObjectDetection
import torch
import matplotlib.pyplot as plt


# COMMAND ----------

TD_THRESHOLD = 0.7
TSR_THRESHOLD = 0.8
delta_xmin = 0
delta_ymin = 0
delta_xmax = 0
delta_ymax = 0
padd_top = 50
padd_left = 50
padd_bottom = 50
padd_right = 20
FILE_PATH = "/Workspace/Repos/arun.wagle@databricks.com/samples/table-transformer/data/Copy of PMC1064076_table_0.jpg"

# COMMAND ----------

# MAGIC %md
# MAGIC # Visualize results

# COMMAND ----------

# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

def plot_results(pil_img, model, scores, labels, boxes):
    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    for score, label, (xmin, ymin, xmax, ymax),c  in zip(scores.tolist(), labels.tolist(), boxes.tolist(), colors):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
        text = f'{model.config.id2label[label]}: {score:0.2f}'
        ax.text(xmin, ymin, text, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    plt.show()
    
    
def plot_results_detection(image, model, scores, labels, boxes, delta_xmin, delta_ymin, delta_xmax, delta_ymax):
    plt.figure(figsize=(16,10))
    plt.imshow(image)
    ax = plt.gca()
    colors = COLORS * 100
    for score, label, (xmin, ymin, xmax, ymax),c  in zip(scores.tolist(), labels.tolist(), boxes.tolist(), colors):       
        xmin, ymin, xmax, ymax = (
            xmin - delta_xmin,
            ymin - delta_ymin,
            xmax + delta_xmax,
            ymax + delta_ymax,
        )
        ax.add_patch(
            plt.Rectangle(
                (xmin, ymin),
                xmax - xmin,
                ymax - ymin,
                fill=False,
                color="red",
                linewidth=3,
            )
        )
        text = f'{model.config.id2label[label]}: {score:0.2f}'
        ax.text(
            xmin - 20,
            ymin - 50,
            text,
            fontsize=10,
            bbox=dict(facecolor="yellow", alpha=0.5),
        )
    plt.axis("off")

# COMMAND ----------

# MAGIC %md
# MAGIC # Table Detection

# COMMAND ----------

# MAGIC %md
# MAGIC Let's first apply the regular image preprocessing using DetrFeatureExtractor. The feature extractor will resize the image (minimum size = 800, max size = 1333), and normalize it across the channels using the ImageNet mean and standard deviation.

# COMMAND ----------

def table_detector(image, THRESHOLD_PROBA):
    """
    Using https://huggingface.co/docs/transformers/main/en/model_doc/table-transformer
    """

    feature_extractor = DetrFeatureExtractor(do_resize=True, size=800, max_size=800)
    encoding = feature_extractor(image, return_tensors="pt")
    encoding.keys()
    print(encoding['pixel_values'].shape)

    model = TableTransformerForObjectDetection.from_pretrained(
        "microsoft/table-transformer-detection"
    )

    with torch.no_grad():
        outputs = model(**encoding)
     
    target_sizes = [image.size[::-1]]
    postprocessed_outputs = feature_extractor.post_process_object_detection(outputs, threshold=THRESHOLD_PROBA, target_sizes=target_sizes)[0]
#     print (postprocessed_outputs)

    return (model, postprocessed_outputs)

# COMMAND ----------

# MAGIC %md
# MAGIC # Table Structure Recognition

# COMMAND ----------

def table_struct_recog(image, THRESHOLD_PROBA):
    """
    Table structure recognition using DEtect-object TRansformer pre-trained on 1 million tables
    """

    feature_extractor = DetrFeatureExtractor(do_resize=True, size=1000, max_size=1000)
    encoding = feature_extractor(image, return_tensors="pt")

    model = TableTransformerForObjectDetection.from_pretrained(
        "microsoft/table-transformer-structure-recognition"
    )
    with torch.no_grad():
        outputs = model(**encoding)

    target_sizes = [image.size[::-1]]
    postprocessed_outputs = feature_extractor.post_process_object_detection(outputs, threshold=THRESHOLD_PROBA, target_sizes=target_sizes)[0]
#     print (postprocessed_outputs)
    
    return (model, postprocessed_outputs)

# COMMAND ----------

# MAGIC %md
# MAGIC # Helper methods
# MAGIC 
# MAGIC ## Crop Table
# MAGIC ## Add Padding

# COMMAND ----------

def crop_tables(pil_img, prob, boxes, delta_xmin, delta_ymin, delta_xmax, delta_ymax):
    """
    crop_tables and plot_results_detection must have same co-ord shifts because 1 only plots the other one updates co-ordinates
    """
    cropped_img_list = []

    for p, (xmin, ymin, xmax, ymax) in zip(prob, boxes.tolist()):

        xmin, ymin, xmax, ymax = (
            xmin - delta_xmin,
            ymin - delta_ymin,
            xmax + delta_xmax,
            ymax + delta_ymax,
        )
        cropped_img = pil_img.crop((xmin, ymin, xmax, ymax))
#         cropped_img.show()
        cropped_img_list.append(cropped_img)
    return cropped_img_list
  
def add_padding(pil_img, top, right, bottom, left, color=(255, 255, 255)):
    """
    Image padding as part of TSR pre-processing to prevent missing table edges
    """
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result

# COMMAND ----------

def process_image(plot_detected_table, plot_detected_table_structure):
  image = Image.open(FILE_PATH).convert("RGB")
  width, height = image.size
  image.resize((int(width*0.5), int(height*0.5)))
  model, results = table_detector(image, THRESHOLD_PROBA=TD_THRESHOLD)
  if plot_detected_table:
    plot_results(image, model, results['scores'], results['labels'], results['boxes'])
  
#   crop the table
  cropped_img_list = crop_tables(image, results['scores'], results['boxes'], delta_xmin, delta_ymin, delta_xmax, delta_ymax)
  
  result = []
  for idx, cropped_image in enumerate(cropped_img_list):
#     cropped_image.show()
    
    padded_image = add_padding(cropped_image, padd_top, padd_right, padd_bottom, padd_left)
#     padded_image.show()
    model, results = table_struct_recog(padded_image, THRESHOLD_PROBA=0.6)
    
    if plot_detected_table_structure:
      
#       plot_results_detection(padded_image, model, results['scores'], results['labels'], results['boxes'], delta_xmin, delta_ymin, delta_xmax, delta_ymax)
      plot_results(padded_image, model, results['scores'], results['labels'], results['boxes'])
  

# COMMAND ----------

process_image (False, True)

# COMMAND ----------

target_sizes = [image.size[::-1]]
results = feature_extractor.post_process_object_detection(outputs, threshold=0.6, target_sizes=target_sizes)[0]
plot_results(image, results['scores'], results['labels'], results['boxes'])

# COMMAND ----------

model.config.id2label

# COMMAND ----------

results

# COMMAND ----------

res = results.get('boxes').numpy()

res
