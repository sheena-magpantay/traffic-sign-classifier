import re

def parse_nlp_feedback(feedback, label_map):

    feedback = feedback.lower()

    # Try exact class name matches
    for class_index, class_name in label_map.items():
        class_name_normal = class_name.lower().replace("_", " ")
        if class_name_normal in feedback:
            return class_index

    # Try extracting numbers
    numbers = re.findall(r'\d+', feedback)
    if numbers:
        for class_index, class_name in label_map.items():
            if any(num in class_name for num in numbers):
                return class_index

    return None