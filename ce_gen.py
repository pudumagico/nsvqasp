import argparse

import clingo

from tools import Context

def main(abduction_program_path, theory_program_path, input_file_path):
    
    abduction_file = open(abduction_program_path)
    abduction = abduction_file.read()
    abduction_file.close()
    theory_file = open(theory_program_path)
    theory = theory_file.read()
    theory_file.close()
    input_file = open(input_file_path)
    input = input_file.read()
    input_file.close()

    ctl_abduction = clingo.Control(['--warn=none', '--opt-strategy=usc'])

    ctl_abduction.add("base", [], abduction)
    ctl_abduction.add("base", [], theory)
    ctl_abduction.add("base", [], input)
    ctl_abduction.assign_external(clingo.String('e_id'), truth=True)
    ctl_abduction.assign_external(clingo.String('e_position'), truth=True)
    ctl_abduction.assign_external(clingo.String('e_size'), truth=True)
    ctl_abduction.assign_external(clingo.String('e_color'), truth=True)
    ctl_abduction.assign_external(clingo.String('e_material'), truth=True)
    ctl_abduction.assign_external(clingo.String('e_shape'), truth=True)

    ctl_abduction.ground([("base", [])], context=Context())
    with ctl_abduction.solve(yield_=True) as handle:
        for model in handle:
            print(model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--abduction_program', type=str, required=True)
    parser.add_argument('--theory_program', type=str, required=True)
    parser.add_argument('--input_file', type=str, required=True)
    #clingo options
    args = parser.parse_args()
    main(args.abduction_program, args.theory_program, args.input_file)
    