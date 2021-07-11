import os
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
gauth = GoogleAuth()

gauth.LoadCredentialsFile('mycreds.txt')
drive = GoogleDrive(gauth)

file_list1 = drive.ListFile({'q': "'17mMAUlPvfQ7EolLVXOq4TxgRdYFLEC73' in parents and trashed=false"}).GetList()
for file1 in file_list1:
    print('title: %s, id: %s' % (file1['title'], file1['id']))

file6 = drive.CreateFile({'id': '1pplYUKWB1fsY2wlIHQaNr9zOn5jI7XKn'})
file6.GetContentFile('data.zip')