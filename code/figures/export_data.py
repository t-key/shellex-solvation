"""Export Fig2/Fig3 tables through the same frozen aggregation entry point."""
from pathlib import Path
import subprocess,sys
if __name__=='__main__':
    root=Path(__file__).resolve().parents[2]
    subprocess.run([sys.executable,str(root/'reproduce_tables.py')],check=True)
    print(root/'results/reproduced/FIG2_COMPUTED_AXIS_MAE.csv')
    print(root/'results/reproduced/FIG3_MEASURED_MAE.csv')
