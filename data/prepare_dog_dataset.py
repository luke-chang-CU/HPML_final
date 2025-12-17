
import os
import csv

def prepare_dog_dataset():
    # 1. Identify Target Classes (Indices 08 to 17)
    train_csv_path = 'dataset/train.csv'
    unique_labels = set()
    rows = []
    
    with open(train_csv_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if len(row) >= 2:
                rows.append(row)
                unique_labels.add(row[1])
                
    sorted_classes = sorted(list(unique_labels))
    
    # Indices 8 to 17 (inclusive? "08 to 17" usually means 8..17, so 10 classes)
    target_classes = sorted_classes[8:18]
    target_class_set = set(target_classes)
    
    print(f"Total Unique Classes: {len(sorted_classes)}")
    print(f"Target Classes (Indices 8-17): {target_classes}")
    
    # 2. Filter Rows
    filtered_rows = [row for row in rows if row[1] in target_class_set]
    
    print(f"Total Rows: {len(rows)}")
    print(f"Filtered Rows (Dogs): {len(filtered_rows)}")
    
    # 3. Write New CSV
    output_path = 'dataset/dogs_08_17.csv'
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(filtered_rows)
        
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    prepare_dog_dataset()
