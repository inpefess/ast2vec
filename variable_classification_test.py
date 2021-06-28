import ast
import python_ast_utils
import ast2vec
import variable_classifier
import astor
import matplotlib.pyplot as plt

## parse an example program
#test_prog = """
#def f(x):
#    return x + 1

#x = f(2)

#print(x)
#"""

#mod = ast.parse(test_prog)
#tree = python_ast_utils.ast_to_tree(mod)

#def print_tree_as_prog(tree):
#    mod2 = python_ast_utils.tree_to_ast(tree)
#    print(astor.to_source(mod2))

#print_tree_as_prog(tree)

## set up a variable classifier
#model = ast2vec.load_model()
#cls = variable_classifier.VariableClassifier(model)

## manually define the needed variables for processing the
## example program
#cls.fun_names_ = ['f']
#cls.cls_fun_name_ = 0
#cls.var_names_ = ['x']
#cls.cls_var_name_ = 0
#cls.cls_fun_ = len(python_ast_utils._builtin_funs)
#cls.cls_var_ = 0
#cls.values_  = [1]
#cls.cls_val_ = 0

## process the example tree
#import copy
#tree_copy = copy.deepcopy(tree)
#cls.predict(tree_copy, True)

## show it
#print_tree_as_prog(tree_copy)

## train the classifier instead
#cls.fit([tree], verbose = True)

## show the result
#cls.predict(tree_copy)
#print_tree_as_prog(tree_copy)




import os

# load the data first. We create a list of all programs and a list of traces,
# where each trace is a list of program indices
programs     = ['']
traces       = []
# start loading the data
student_dirs = list(sorted(os.listdir('mock_dataset')))
for student_dir in student_dirs:
    # initialize a new trace for the student which starts at the empty program
    trace = [0]
    # load all steps of this student
    steps = list(sorted(os.listdir(f'mock_dataset/{student_dir}')))
    for step in steps:
        # load the current program
        with open(f'mock_dataset/{student_dir}/{step}') as program_file:
            trace.append(len(programs))
            programs.append(program_file.read())
    # append the trace
    traces.append(trace)

trees, programs_to_trees = python_ast_utils.parse_asts(programs, filter_uniques = True)

# check if all programs can be formatted back as asts
for tree in trees:
    try:
        ast_tree = python_ast_utils.tree_to_ast(tree)
        astor.to_source(ast_tree)
    except Exception as ex:
        print(tree.pretty_print())
        raise ex

# now we load the ast2vec model
model = ast2vec.load_model()

# and encode all the syntax trees as points
X = ast2vec.encode_trees(model, trees)

# initialize a variable classifier
vc = variable_classifier.VariableClassifier(model)
# fit it to our example dataset
vc.fit(trees, verbose = True)

# get the index of the program, which is the last program in the first trace
program_index = traces[0][-1]
# get the corresponding syntax tree index
tree_index    = programs_to_trees[program_index]

tree_traces = [[0] + [programs_to_trees[program] for program in trace] for trace in traces]


# perform a linear interpolation on a grid between empty program and correct solution


plt.figure(figsize = (15, 10))
ast2vec.interpolation_plot(model, start_tree = 0, target_tree = tree_index, X = X, variable_classifier = vc)
plt.show()

# construct the dynamical system via linear regression
W = ast2vec.construct_dynamical_system(tree_index, X, tree_traces)
# visualize the resulting dynamical system
plt.figure(figsize = (15, 10))
ast2vec.dynamical_system_plot(W, start_tree = 0, target_tree = tree_index, X = X, arrow_scale = 1.5, model = model, variable_classifier = vc)
plt.show()
