# This code is part of Rice COMP182 and is made available for your
# use as a student in COMP182. You are specifically forbidden from
# posting this code online in a public fashion (e.g., on a public
# GitHub repository) or otherwise making it, or any derivative of it,
# available to future COMP182 students.

import traceback
import re
import numpy
from collections import defaultdict

print("""DISCLAIMER: This tool is intended to ensure your code is 
compatible with the autograder, not that it is correct. It is 
possible to 'pass' this tool, yet receive a 0 on the coding portion. 
You are still responsible for writing your own tests to ensure the 
correctness of your code.
""")

class SkeletonAutograder():
    def __init__(self):
        self._allowed_imports = []
        self._test_cases_functions = []
        self._test_cases_inputs = []
        self._test_cases_expected = []

    def set_allowed_imports(self, imports: list):
        self._allowed_imports = imports

    def add_test_case(self, function, inputs: list, outputs: list):
        self._test_cases_functions.append(function)
        self._test_cases_inputs.append(inputs)
        self._test_cases_expected.append(outputs)

    def fail_test(self):
        print("\nFAILED!")
        exit()

    def check_imports(self):
        """
        This method verifies that only allowed imports are used in 
        student's submission. Requires 'set_allowed_imports' to be 
        executed before checking.

        If there is an illegal import, then it fails and exits the 
        skeleton autograder.
        """

        # Set regular expression to match Python import statements.
        pattern = re.compile(r"(^from\s(\w+)\simport\s([\w*]+)$)|(^import\s(\w+)$)")

        # Define list for illegally used imports.
        illegal_imports = []

        with open("autograder.py") as f:
            lines = f.readlines()
            for line in lines:
                # Match the pattern.
                line = re.sub(r'\s+$', '', re.sub(r'^\s+', '', line))
                match = pattern.match(line)

                # Check for matches.
                if match is not None:
                    groups = match.groups(default='')
                    importstr = " ".join(groups[1:3] if groups[0] else [groups[4]])
                    if importstr not in self._allowed_imports:
                        illegal_imports.append(line)

        if len(illegal_imports) > 0:
            print("A disallowed import was detected. Please remove this import and re-run the autograder.\nThe line(s) in question are:")
            for line in illegal_imports:
                print(line)

            self.fail_test()

    def check_directory(self):
        """
        This method verifies that student submission is in the same directory as the skeleton autograder.

        If the skeleton autograder cannot import 'autograder.py', then it fails and exists the skeleton autograder.
        """
        try:
            import autograder
        except ImportError as e:
            print("""Failed to import 'autograder.py'.
            Ensure the following:
                1. Your submission is titled 'autograder'.py
                2. Your submitted 'autograder.py' file is in the same directory as this file ('skeleton_autograder.py')
                3. Your submission doesn't import anything other than the imports in the original provided template file
            See the error below for more information:\n"""+traceback.format_exc())

            self.fail_test()

        except Exception as e:
            print("""Failed to import 'autograder.py'.
            Your code likely failed due to code located outside a function failing.
            Ensure the following:
                1. All of your code is in one of the autograder or helper functions
                2. Any testing code, or code outside of a function, is commented out
            See the error below for more information:\n"""+traceback.format_exc())

            self.fail_test()

    def run_tests(self, run_typechecks = False):
        """
        This method runs all the test cases defined. By default, it checks whether the autograder.py is located in
        the same directory and whether imports are legal.

        Flag run_typechecks can be toggled to 
        """

        # Run default tests to ensure form.
        self.check_directory()
        self.check_imports()

        import autograder

        for test_id, func_name in enumerate(self._test_cases_functions):

            # Try to get the function, if the function cannot be located in autograder, then fail the test.
            try:
                func = getattr(autograder, func_name)
            except AttributeError:
                print("Could not locate function '" + func_name + "', ensure your code contains a function with that exact name.")
                print("See the error below for more information:\n")
                print(traceback.format_exc())

                self.fail_test()

            inputs = self._test_cases_inputs[test_id]
            expected = self._test_cases_expected[test_id]

            # Run student's function.
            print("Running Test #"+str(test_id)+" on '"+ func_name + "'...")
            try:
                actual = func(*inputs)

                print("Input(s): "+ str(inputs))
                print("Expected Output(s): "+ str(expected))
                print("Actual Output(s)  : "+ str(actual))
                print("")

                if run_typechecks and type(expected) is not type(actual):
                    print("Wrong type returned, expecting '" + str(type(expected)) + "', received '" + str(type(actual)) + "'.")

                    self.fail_test()

                if type(expected) == list or type(expected) == tuple:
                    if len(expected) != len(actual):
                        print("Was expecting "+str(len(expected))+" number of output, received "+str(len(actual))+".")

                        self.fail_test()

                if expected != actual:
                    print("Wrong value returned, expecting '" + str(expected) + ", received '" + str(actual) + "'.")

                    self.fail_test()

                print("Test passed!\n")

            except Exception as e:
                print("Code failed to run, see the error below for more information:\n")
                print(traceback.format_exc())
                
                self.fail_test()
            


skeleton_autograder = SkeletonAutograder()
skeleton_autograder.set_allowed_imports(['random', 'collections *'])

## Homework 4 Test Cases
graph_one = {"a": set(["b","c"]), "b": set(["a"]), "c": set(["a"])}

bfs_graph_one = tuple([graph_one, "a"])
bfs_graph_one_sols = ({"a": 0, "b": 1, "c": 1},{"a": 1, "b": 1, "c": 1})

compute_flow_graph_one_sols = {frozenset(["a","b"]): 1, frozenset(["a","c"]): 1}
shortest_path_edge_betweenness_sols = {frozenset(["a","b"]): 4, frozenset(["a","c"]): 4}
compute_q_communities = [set(["a","b"]),set("c")]
compute_q_sols = -0.75

skeleton_autograder.add_test_case("compute_flow", [graph_one, bfs_graph_one_sols[0], bfs_graph_one_sols[1]], compute_flow_graph_one_sols)
skeleton_autograder.add_test_case("shortest_path_edge_betweenness", [graph_one], shortest_path_edge_betweenness_sols)
skeleton_autograder.add_test_case("compute_q", [graph_one, compute_q_communities], compute_q_sols)

skeleton_autograder.run_tests(run_typechecks = False)
