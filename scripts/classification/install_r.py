import rpy2
from rpy2.robjects.packages import importr, isinstalled

def install_R_packages(packnames):
    utils = importr('utils')
    utils.install_packages(rpy2.robjects.StrVector(packnames))

if __name__ == '__main__':
    install = ['rstatix', 'effectsize', 'broom', 'knitr']
    install = [i for i in install if not isinstalled(i)]

    if len(install) != 0: 
        print(f"R packages to be installed: {install}")
        print("WARNING: This may take awhile")
        install_R_packages(install)
    else:
        print("No packages to install currently.")

