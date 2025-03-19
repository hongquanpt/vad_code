# https://realpython.com/python-csv/
import csv, pandas

def openfile_csv(file_path = 'file.csv'):
    '''
    Reading CSV Files With csv
    '''
    with open(file_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            str_row = ""
            for item in row:
                str_row += item
            print(str_row)

def openfile_csv_dic(file_path = 'file.csv'):
    '''
    Reading CSV Files Into a Dictionary With csv
    '''
    with open(file_path) as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            str_row = ""
            for item in row:
                str_row += item
            print(str_row)

def writefile_csv_dic(file_path = 'file.csv', 
                      mode = 'w',
                      fieldnames =  ['ID', 'Value'], 
                      row = {'ID': '1', 'Value': '0001.jpg'}):
    '''
    Writing CSV File From a Dictionary With csv
    '''
    with open(file_path, mode=mode) as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if mode == 'w':
            writer.writeheader()
            print(f'Created a new file: {file_path}')
        writer.writerow(row)
    
def openfile_pandas(file_path = 'file.csv'):
    '''
    Reading CSV Files With pandas
    '''
    df = pandas.read_csv(file_path)
    print(df)
            
