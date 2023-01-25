from attack import train_normal, train_generator, train_compare, evaluate
from config import parse_param
from defense import test_defense
from tools import send_complete_notification, ATTACKER


def main():
    opt = parse_param()
    command = opt.command
    if command == 'n':
        train_normal(opt)
    elif command == 'a':
        if opt.attacker == ATTACKER.generator:
            train_generator(opt)
        else:
            train_compare(opt)
    elif command == 'e':
        evaluate(opt)
    elif command in ['nc', 'sp', 'fp']:
        test_defense(opt)
    else:
        raise ValueError(f'command should in [n, a, e, nc, sp, fp], not {command}')
    if command in ['n', 'a', 'nc', 'fp']:
        send_complete_notification()


if __name__ == '__main__':
    main()
