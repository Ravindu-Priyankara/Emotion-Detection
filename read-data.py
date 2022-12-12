#!/ user/bin/python3

class Main:
    def neutral_data():
        file = open('data-of-neutral.txt','r')
        count = 0

        # line = file.readline()
        while True:
            count +=1
            line = file.readline()

            if not line:
                return count
                break
        file.close()
    def fear_data():
        pass
    def 

if __name__ == '__main__':
    test = Main
    #test.test01()
    print(test.test01())