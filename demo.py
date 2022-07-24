from energy_efficiency.component.get_data import DataCollection
from energy_efficiency.component.do_validation import DataValidation
from energy_efficiency.component.data_transform import DataTransformation
def main():
    d=DataCollection()
    d.get_data_from_database()
    print(d.dumping_data())
    v=DataValidation()
    print(v.validate_data())
    D=DataTransformation()
    print(D.transform_data())

main()

