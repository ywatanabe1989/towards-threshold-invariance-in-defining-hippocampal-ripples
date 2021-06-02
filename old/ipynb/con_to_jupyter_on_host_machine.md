## Host
$ jupyter-notebook --generate-config

c.NotebookApp.ip = 'localhost'
c.NotebookApp.open_browser = False
c.NotebookApp.port = 8889
c.NotebookApp.token = u''

jupyter-notebook password

{
  "NotebookApp": {
    "password": "#####"
  }
}

c.NotebookApp.password = '#####'


## Client
ssh -L 8889:localhost:8889 example.com
https://localhost:8889/
