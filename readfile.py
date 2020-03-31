import csv
def create_csv(in_route, head_list):
    path = in_route
    with open(path,'wb') as f:
        csv_write = csv.writer(f)
        csv_head = head_list
        csv_write.writerow(csv_head)

def write_raw_index(file, head_list):
    filename = file
    with open(filename, 'r+') as f:
        content = f.read()
        f.seek(0, 0)
        #mid, text, source, uid
        text = head_list
        f.write(text + '\n' + content)