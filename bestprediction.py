# bestpredictions.py

def calculate_iou(box1, box2):
    # Function to calculate Intersection over Union (IoU) between two bounding boxes
    # Each box is represented as a list or tuple of [x_min, y_min, x_max, y_max]
    x_min1, y_min1, x_max1, y_max1 = box1
    x_min2, y_min2, x_max2, y_max2 = box2

    # Calculate the coordinates of the intersection rectangle
    inter_x_min = max(x_min1, x_min2)
    inter_y_min = max(y_min1, y_min2)
    inter_x_max = min(x_max1, x_max2)
    inter_y_max = min(y_max1, y_max2)

    # Calculate the area of the intersection rectangle
    inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)

    # Calculate the area of both the prediction and ground truth rectangles
    box1_area = (x_max1 - x_min1) * (y_max1 - y_min1)
    box2_area = (x_max2 - x_min2) * (y_max2 - y_min2)

    # Calculate the Intersection over Union (IoU)
    iou = inter_area / float(box1_area + box2_area - inter_area)

    return iou

def select_best_prediction(predictions, iou_threshold=0.5):
    n = len(predictions)
    groups = []  # List to hold groups of indices of competitive predictions
    no_overlap_indices = set(range(n))  # Indices of predictions with no overlaps

    # Step 1: Determine competitive predictions based on IoU
    for i in range(n):
        for j in range(i + 1, n):
            iou = calculate_iou(predictions[i]['box'], predictions[j]['box'])
            if iou > iou_threshold:
                no_overlap_indices.discard(i)
                no_overlap_indices.discard(j)
                found_group = False
                # Try to add these predictions' indices to an existing group
                for group in groups:
                    if i in group or j in group:
                        group.update([i, j])
                        found_group = True
                        break
                # If no existing group is found, create a new group with indices
                if not found_group:
                    groups.append(set([i, j]))

    # Step 2: Handle predictions with no overlaps
    best_predictions = [predictions[i] for i in no_overlap_indices]

    # Step 3: Merge overlapping groups to handle multiple overlaps
    merged_groups = []
    while groups:
        first, *rest = groups
        first = set(first)

        lf = -1
        while len(first) > lf:
            lf = len(first)

            rest2 = []
            for r in rest:
                if first & set(r):
                    first |= set(r)
                else:
                    rest2.append(r)
            rest = rest2

        merged_groups.append(first)
        groups = rest

    # Step 4: Select the best prediction from each merged group based on the highest score
    for group in merged_groups:
        best_pred_index = max(group, key=lambda x: predictions[x]['score'])
        best_predictions.append(predictions[best_pred_index])

    return best_predictions
