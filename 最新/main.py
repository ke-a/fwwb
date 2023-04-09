# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import pymysql

def print_hi():
    connect = pymysql.connect(user='root',password='b1b1',db='mysql',charset='utf8')
    sql = "SELECT * From micro1"
    
    cursor = connect.cursor()
    cursor.execute(sql)
    result = cursor.fetchone()
    while result != None:
        print(result)
        result = cursor.fetchone()





# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
