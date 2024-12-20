class Person:
    # The method that runs as soon as you create a class - initialize
    def __init__(self, name, age, color):  # self = refer to this particular object
        # Create some attributes for our person class
        self.name = name
        self.age = age
        self.color = color

    # Date of Birth Method
    def year_of_birth(
        self,
    ):  # pass in self so the method and access to the person attriibutes (name age, color)
        return 2024 - self.age

    # projected age
    def project_age(self, years=5):
        return self.age + years


# inheritance
class Astronaut(Person):
    # define initialization funcion
    def __init__(self, name, age, color, mission_length):
        # this is what gives us inheritance - SUPER METHOD
        super().__init__(name, age, color)
        self.mission_length = mission_length

    # method for calculation return age
    def age_on_return(self):
        # accesing method from parent class
        return self.project_age(years=int(self.mission_length / 12))


# creating a new person
new_person = Person("Guya", 9, "Gold")

# accessing a class attribute
new_person.name

# run a method
birth_year = new_person.year_of_birth()

# run a method
projected_age = new_person.project_age(years=10)

new_astronaut = Astronaut("Guya", 9, "Gold", 48)
return_age = new_astronaut.age_on_return()

print(return_age)
