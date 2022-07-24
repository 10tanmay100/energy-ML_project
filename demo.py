import psutil
from energy_efficiency.component.get_data import DataCollection
from energy_efficiency.component.do_validation import DataValidation
from energy_efficiency.component.data_transform import DataTransformation
from energy_efficiency.component.model_training import model_training
from energy_efficiency.component.model_pusher import pusher
def main():
    d=DataCollection()
    d.get_data_from_database()
    print(d.dumping_data())
    v=DataValidation()
    print(v.validate_data())
    D=DataTransformation()
    print(D.transform_data())
    m=model_training()
    m.splitting_the_data()
    print(m.model_trainer())
    x=pusher()
    print(x.deploy())

main()

