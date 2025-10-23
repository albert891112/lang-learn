from pydantic import BaseModel


class Albert(BaseModel):
    yoyo: str

    def __repr__(self):
        return f"{self.__class__.__name__}({self.__dict__})"

    def __str__(self):
        # 讓 print(person) 使用你自訂的表示法
        return self.__repr__()

    def __init__(self, name: str):
        super().__init__(yoyo=name)


person = Albert("承儒")  # 传入 "承儒" 作为 name 参数
print(person)
