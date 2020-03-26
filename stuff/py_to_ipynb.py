import IPython.nbformat.current as nbf
import os
for root, dirs, files in os.walk(os.getcwd()):
    for file in files:
        if file.endswith(".py"):
            fn = os.path.splitext(file)[0]
            nb = nbf.read(open(file, 'r'), 'py')
            nbf.write(nb, open((fn+'.ipynb'), 'w'), 'ipynb')
