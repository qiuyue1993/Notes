import json

##-------------------------------------------------------------------
## binary
verify_color = "verify color"
verify_rel = "verify rel"
verify_material = "verify material"
verify_size = "verify size"
verify_shape = "verify shape"

same_color = "same color"
same_material = "same material"
same_shape = "same shape"


## unary
filter_color = "filter color"
filter_material = "filter material"
filter_size = "filter size"
filter_shape = "filter shape"

select = "select"
verify = "verify"
same = "same"
relate = "relate"
query = "query"
exist = "exist"
or_ = "or"
and_ = "and"
filter_ = "filter"


operation_dict = {}
operation_dict[verify_color] = len(operation_dict)
operation_dict[verify_rel] = len(operation_dict)
operation_dict[verify_material] = len(operation_dict)
operation_dict[verify_size] = len(operation_dict)
operation_dict[verify_shape] = len(operation_dict)

operation_dict[same_color] = len(operation_dict)
operation_dict[same_material] = len(operation_dict)
operation_dict[same_shape] = len(operation_dict)

operation_dict[filter_color] = len(operation_dict)
operation_dict[filter_material] = len(operation_dict)
operation_dict[filter_size] = len(operation_dict)
operation_dict[filter_shape] = len(operation_dict)

operation_dict[select] = len(operation_dict)


operation_dict[verify] = len(operation_dict)
operation_dict[relate] = len(operation_dict)
operation_dict[query] = len(operation_dict)
operation_dict[exist] = len(operation_dict)
operation_dict[or_] = len(operation_dict)
operation_dict[and_] = len(operation_dict)
operation_dict[filter_] = len(operation_dict)
##---------------------------------------------------------------------


with open('train_nb_questions.json', 'r') as f:
  tq = json.load(f)

with open('train_scene_object_list.json', 'r') as f:
  tol = json.load(f)

with open('val_nb_questions.json', 'r') as f:
  vq = json.load(f)

with open('val_scene_object_list.json', 'r') as f:
  vol = json.load(f)


tqnew = {}
vqnew = {}
tqnew_list = []
vqnew_list = []

#######################
# For attribute names
with open('attris/attribute_vocabulary.json', 'r') as f:
  attribute_names = json.load(f)


with open('attris/relations_vocabulary.json', 'r') as f:
  relations_names = json.load(f)


#######################  tq  tqnew_list

### Modify question one by one
for question_idx in tq.keys():
  current_question = {}
 
  current_question['image_idx'] = tq[question_idx]['imageId']
  current_question['question_idx'] = len(tqnew_list)
  current_question['question_family_index'] = 0
  current_question['question'] = tq[question_idx]['question']
  current_question['answer'] = tq[question_idx]['answer']


  program = []
  scene = {}
  scene['inputs'] = []
  scene['function'] = "scene"
  scene['value_inputs'] = []  
  program.append(scene)

  curr_depend = []
  curr_depend.append([])
  
  iti = 0
  for semantic in tq[question_idx]['semantic']:
    iti = iti + 1
    curr_depend.append([])
    curr_depend[iti] = semantic['dependencies']


  countc = 0
  for semantic in tq[question_idx]['semantic']:
    print("ddd")
 
    countc = countc + 1
    if semantic['operation'] == "verify color":
      print("verify color")
      pg0 = {}
      pg1 = {}

      if curr_depend[countc] == []:
        pg0['inputs'].append([])       
      elif curr_depend[curr_depend[countc][0]] == []:
        pg0['inputs'].append([])
      else:
        pg0['inputs'].append(curr_depend[curr_depend[countc][0]][-1])

      pg0['function'] = "filter_"
      pg0['value_inputs'] = []
      pg0['value_inputs'].append(semantic['argument'])
      program.append(pg0)

      pg1['inputs'] = len(program) - 1
      pg1['function'] = "exist"
      pg1['value_inputs'] = []
      program.append(pg1) 
      
      curr_depend[countc].append(len(program)-1)

      
    elif semantic['operation'] == "verify rel":
      pg0 = {}
      pg1 = {}
      pg2 = {}

      if curr_depend[countc] == []:
        pg0['inputs'].append([])       
      elif curr_depend[curr_depend[countc][0]] == []:
        pg0['inputs'].append([])
      else:
        pg0['inputs'].append(curr_depend[curr_depend[countc][0]][-1])

      pg0['function'] = "relate"
      pg0['value_inputs'] = []
      if semantic['argument'].split(',')[1] == "to the left of":
        pg0['value_inputs'].append("left")
      elif semantic['argument'].split(',')[1] == "to the right of":
        pg0['value_inputs'].append("right")
      elif semantic['argument'].split(',')[1] == "in front of":
        pg0['value_inputs'].append("front")
      else:
        pg0['value_inputs'].append(semantic['argument'].split(',')[1])
      program.append(pg0)

      pg1['inputs'] = len(program) - 1
      pg1['function'] = "filter_"
      pg1['value_inputs'] = []
      pg0['value_inputs'].append(semantic['argument'].split(',')[0])
      program.append(pg1) 

      pg2['inputs'] = len(program) - 1
      pg2['function'] = "exist"
      pg2['value_inputs'] = []
      program.append(pg2) 

      curr_depend[countc].append(len(program)-1)

    elif semantic['operation'] == "verify material":
      pg0 = {}
      pg1 = {}

      if curr_depend[countc] == []:
        pg0['inputs'].append([])       
      elif curr_depend[curr_depend[countc][0]] == []:
        pg0['inputs'].append([])
      else:
        pg0['inputs'].append(curr_depend[curr_depend[countc][0]][-1])

      pg0['function'] = "filter_"
      pg0['value_inputs'] = []
      pg0['value_inputs'].append(semantic['argument'])
      program.append(pg0)

      pg1['inputs'] = len(program) - 1
      pg1['function'] = "exist"
      pg1['value_inputs'] = []
      program.append(pg1) 

      curr_depend[countc].append(len(program)-1)

    elif semantic['operation'] == "verify size":
      pg0 = {}
      pg1 = {}

      if curr_depend[countc] == []:
        pg0['inputs'].append([])       
      elif curr_depend[curr_depend[countc][0]] == []:
        pg0['inputs'].append([])
      else:
        pg0['inputs'].append(curr_depend[curr_depend[countc][0]][-1])

      pg0['function'] = "filter_"
      pg0['value_inputs'] = []
      pg0['value_inputs'].append(semantic['argument'])
      program.append(pg0)

      pg1['inputs'] = len(program) - 1
      pg1['function'] = "exist"
      pg1['value_inputs'] = []
      program.append(pg1) 

      curr_depend[countc].append(len(program)-1)

    elif semantic['operation'] == "verify shape":
      pg0 = {}
      pg1 = {}

      if curr_depend[countc] == []:
        pg0['inputs'].append([])       
      elif curr_depend[curr_depend[countc][0]] == []:
        pg0['inputs'].append([])
      else:
        pg0['inputs'].append(curr_depend[curr_depend[countc][0]][-1])

      pg0['function'] = "filter_"
      pg0['value_inputs'] = []
      pg0['value_inputs'].append(semantic['argument'])
      program.append(pg0)

      pg1['inputs'] = len(program) - 1
      pg1['function'] = "exist"
      pg1['value_inputs'] = []
      program.append(pg1) 

      curr_depend[countc].append(len(program)-1)

    elif semantic['operation'] == "same color":
      pg0 = {}

      if curr_depend[countc] == []:
        pg0['inputs'].append([])       
      elif curr_depend[curr_depend[countc][0]] == []:
        pg0['inputs'].append([])
      else:
        pg0['inputs'].append(curr_depend[curr_depend[countc][0]][-1])

      pg0['function'] = "equal"
      pg0['value_inputs'] = []
      pg0['value_inputs'].append("color")
      program.append(pg0)

      curr_depend[countc].append(len(program)-1)

    elif semantic['operation'] == "same material":
      pg0 = {}

      if curr_depend[countc] == []:
        pg0['inputs'].append([])       
      elif curr_depend[curr_depend[countc][0]] == []:
        pg0['inputs'].append([])
      else:
        pg0['inputs'].append(curr_depend[curr_depend[countc][0]][-1])

      pg0['function'] = "equal"
      pg0['value_inputs'] = []
      pg0['value_inputs'].append("material")
      program.append(pg0)

      curr_depend[countc].append(len(program)-1)

    elif semantic['operation'] == "same shape":
      pg0 = {}

      if curr_depend[countc] == []:
        pg0['inputs'].append([])       
      elif curr_depend[curr_depend[countc][0]] == []:
        pg0['inputs'].append([])
      else:
        pg0['inputs'].append(curr_depend[curr_depend[countc][0]][-1])

      pg0['function'] = "equal"
      pg0['value_inputs'] = []
      pg0['value_inputs'].append("shape")
      program.append(pg0)

      curr_depend[countc].append(len(program)-1)

    elif semantic['operation'] == "filter color":
      pg0 = {}

      if curr_depend[countc] == []:
        pg0['inputs'].append([])       
      elif curr_depend[curr_depend[countc][0]] == []:
        pg0['inputs'].append([])
      else:
        pg0['inputs'].append(curr_depend[curr_depend[countc][0]][-1])

      pg0['function'] = "filter_"
      pg0['value_inputs'] = []
      pg0['value_inputs'].append(semantic['argument'])
      program.append(pg0)

      curr_depend[countc].append(len(program)-1)

    elif semantic['operation'] == "filter material":
      pg0 = {}

      if curr_depend[countc] == []:
        pg0['inputs'].append([])       
      elif curr_depend[curr_depend[countc][0]] == []:
        pg0['inputs'].append([])
      else:
        pg0['inputs'].append(curr_depend[curr_depend[countc][0]][-1])

      pg0['function'] = "filter_"
      pg0['value_inputs'] = []
      pg0['value_inputs'].append(semantic['argument'])
      program.append(pg0)

      curr_depend[countc].append(len(program)-1)

    elif semantic['operation'] == "filter shape":
      pg0 = {}

      if curr_depend[countc] == []:
        pg0['inputs'].append([])       
      elif curr_depend[curr_depend[countc][0]] == []:
        pg0['inputs'].append([])
      else:
        pg0['inputs'].append(curr_depend[curr_depend[countc][0]][-1])

      pg0['function'] = "filter_"
      pg0['value_inputs'] = []
      pg0['value_inputs'].append(semantic['argument'])
      program.append(pg0)

      curr_depend[countc].append(len(program)-1)

    elif semantic['operation'] == "filter size":
      pg0 = {}

      if curr_depend[countc] == []:
        pg0['inputs'].append([])       
      elif curr_depend[curr_depend[countc][0]] == []:
        pg0['inputs'].append([])
      else:
        pg0['inputs'].append(curr_depend[curr_depend[countc][0]][-1])


      pg0['function'] = "filter_"
      pg0['value_inputs'] = []
      pg0['value_inputs'].append(semantic['argument'])
      program.append(pg0)

      curr_depend[countc].append(len(program)-1)


    elif semantic['operation'] == "select":
      pg0 = {}

      pg0['inputs'] = []
 
      if curr_depend[countc] == []:
        pg0['inputs'].append([])       
      elif curr_depend[curr_depend[countc][0]] == []:
        pg0['inputs'].append([])
      else:
        pg0['inputs'].append(curr_depend[curr_depend[countc][0]][-1])

      pg0['function'] = "filter_"
      pg0['value_inputs'] = []
      pg0['value_inputs'].append(semantic['argument'].split('(')[0])
      program.append(pg0)

      curr_depend[countc].append(len(program)-1)

    elif semantic['operation'] == "relate":
      pg0 = {}

      if curr_depend[countc] == []:
        pg0['inputs'].append([])       
      elif curr_depend[curr_depend[countc][0]] == []:
        pg0['inputs'].append([])
      else:
        pg0['inputs'].append(curr_depend[curr_depend[countc][0]][-1])


      pg0['function'] = "relate"
      pg0['value_inputs'] = []
      if semantic['argument'].split(',')[1] == "to the left of":
        pg0['value_inputs'].append("left")
      elif semantic['argument'].split(',')[1] == "to the right of":
        pg0['value_inputs'].append("right")
      elif semantic['argument'].split(',')[1] == "in front of":
        pg0['value_inputs'].append("front")
      else:
        pg0['value_inputs'].append(semantic['argument'].split(',')[1])
      program.append(pg0)

      curr_depend[countc].append(len(program)-1)

    elif semantic['operation'] == "query":
      pg0 = {}

      if curr_depend[countc] == []:
        pg0['inputs'].append([])       
      elif curr_depend[curr_depend[countc][0]] == []:
        pg0['inputs'].append([])
      else:
        pg0['inputs'].append(curr_depend[curr_depend[countc][0]][-1])


      pg0['function'] = "query"
      pg0['value_inputs'] = []
      pg0['value_inputs'].append(semantic['argument'])
      program.append(pg0)

      curr_depend[countc].append(len(program)-1)


    elif semantic['operation'] == "exist":
      pg0 = {}

      if curr_depend[countc] == []:
        pg0['inputs'].append([])       
      elif curr_depend[curr_depend[countc][0]] == []:
        pg0['inputs'].append([])
      else:
        pg0['inputs'].append(curr_depend[curr_depend[countc][0]][-1])

      pg0['function'] = "exist"
      pg0['value_inputs'] = []
      program.append(pg0)

      curr_depend[countc].append(len(program)-1)

    elif semantic['operation'] == "or":  ### ----------------------
      pg0 = {}

      if curr_depend[countc] == []:
        pg0['inputs'].append([])       
      elif curr_depend[curr_depend[countc][0]] == []:
        pg0['inputs'].append([])
      else:
        pg0['inputs'].append(curr_depend[curr_depend[countc][0]][-1])
      pg0['function'] = "union"
      pg0['value_inputs'] = []
      program.append(pg0)

      curr_depend[countc].append(len(program)-1)

    elif semantic['operation'] == "and":  ### ----------------------
      pg0 = {}

      if curr_depend[countc] == []:
        pg0['inputs'].append([])       
      elif curr_depend[curr_depend[countc][0]] == []:
        pg0['inputs'].append([])
      else:
        pg0['inputs'].append(curr_depend[curr_depend[countc][0]][-1])
      pg0['function'] = "intersect"
      pg0['value_inputs'] = []
      program.append(pg0)

      curr_depend[countc].append(len(program)-1)

    elif semantic['operation'] == "filter":
      pg0 = {}

      if curr_depend[countc] == []:
        pg0['inputs'].append([])       
      elif curr_depend[curr_depend[countc][0]] == []:
        pg0['inputs'].append([])
      else:
        pg0['inputs'].append(curr_depend[curr_depend[countc][0]][-1])

      pg0['function'] = "filter_"
      pg0['value_inputs'] = []
      pg0['value_inputs'].append(semantic['argument'])
      program.append(pg0)

      curr_depend[countc].append(len(program)-1)

    else:
      print(semantic['operation'])

  current_question['program'] = program
  tqnew_list.append(current_question)



### Modify question one by one
for question_idx in vq.keys():
  current_question = {}

  current_question['image_idx'] = vq[question_idx]['imageId']
  current_question['question_idx'] = len(vqnew_list)
  current_question['question_family_index'] = 0
  current_question['question'] = vq[question_idx]['question']
  current_question['answer'] = vq[question_idx]['answer']

  program = []
  scene = {}
  scene['inputs'] = []
  scene['function'] = "scene"
  scene['value_inputs'] = []  
  program.append(scene)

  curr_depend = []
  curr_depend.append([])
  
  iti = 0
  for semantic in vq[question_idx]['semantic']:
    iti = iti + 1
    curr_depend.append([])
    curr_depend[iti] = semantic['dependencies']


  countc = 0
  for semantic in vq[question_idx]['semantic']:
    countc = countc + 1
    if semantic['operation'] == "verify color":
      pg0 = {}
      pg1 = {}

      if curr_depend[countc] == []:
        pg0['inputs'].append([])       
      elif curr_depend[curr_depend[countc][0]] == []:
        pg0['inputs'].append([])
      else:
        pg0['inputs'].append(curr_depend[curr_depend[countc][0]][-1])

      pg0['function'] = "filter_"
      pg0['value_inputs'] = []
      pg0['value_inputs'].append(semantic['argument'])
      program.append(pg0)

      pg1['inputs'] = len(program) - 1
      pg1['function'] = "exist"
      pg1['value_inputs'] = []
      program.append(pg1) 
      
      curr_depend[countc].append(len(program)-1)

      
    elif semantic['operation'] == "verify rel":
      pg0 = {}
      pg1 = {}
      pg2 = {}

      if curr_depend[countc] == []:
        pg0['inputs'].append([])       
      elif curr_depend[curr_depend[countc][0]] == []:
        pg0['inputs'].append([])
      else:
        pg0['inputs'].append(curr_depend[curr_depend[countc][0]][-1])

      pg0['function'] = "relate"
      pg0['value_inputs'] = []
      if semantic['argument'].split(',')[1] == "to the left of":
        pg0['value_inputs'].append("left")
      elif semantic['argument'].split(',')[1] == "to the right of":
        pg0['value_inputs'].append("right")
      elif semantic['argument'].split(',')[1] == "in front of":
        pg0['value_inputs'].append("front")
      else:
        pg0['value_inputs'].append(semantic['argument'].split(',')[1])
      program.append(pg0)

      pg1['inputs'] = len(program) - 1
      pg1['function'] = "filter_"
      pg1['value_inputs'] = []
      pg0['value_inputs'].append(semantic['argument'].split(',')[0])
      program.append(pg1) 

      pg2['inputs'] = len(program) - 1
      pg2['function'] = "exist"
      pg2['value_inputs'] = []
      program.append(pg2) 

      curr_depend[countc].append(len(program)-1)

    elif semantic['operation'] == "verify material":
      pg0 = {}
      pg1 = {}

      if curr_depend[countc] == []:
        pg0['inputs'].append([])       
      elif curr_depend[curr_depend[countc][0]] == []:
        pg0['inputs'].append([])
      else:
        pg0['inputs'].append(curr_depend[curr_depend[countc][0]][-1])

      pg0['function'] = "filter_"
      pg0['value_inputs'] = []
      pg0['value_inputs'].append(semantic['argument'])
      program.append(pg0)

      pg1['inputs'] = len(program) - 1
      pg1['function'] = "exist"
      pg1['value_inputs'] = []
      program.append(pg1) 

      curr_depend[countc].append(len(program)-1)

    elif semantic['operation'] == "verify size":
      pg0 = {}
      pg1 = {}

      if curr_depend[countc] == []:
        pg0['inputs'].append([])       
      elif curr_depend[curr_depend[countc][0]] == []:
        pg0['inputs'].append([])
      else:
        pg0['inputs'].append(curr_depend[curr_depend[countc][0]][-1])

      pg0['function'] = "filter_"
      pg0['value_inputs'] = []
      pg0['value_inputs'].append(semantic['argument'])
      program.append(pg0)

      pg1['inputs'] = len(program) - 1
      pg1['function'] = "exist"
      pg1['value_inputs'] = []
      program.append(pg1) 

      curr_depend[countc].append(len(program)-1)

    elif semantic['operation'] == "verify shape":
      pg0 = {}
      pg1 = {}

      if curr_depend[countc] == []:
        pg0['inputs'].append([])       
      elif curr_depend[curr_depend[countc][0]] == []:
        pg0['inputs'].append([])
      else:
        pg0['inputs'].append(curr_depend[curr_depend[countc][0]][-1])

      pg0['function'] = "filter_"
      pg0['value_inputs'] = []
      pg0['value_inputs'].append(semantic['argument'])
      program.append(pg0)

      pg1['inputs'] = len(program) - 1
      pg1['function'] = "exist"
      pg1['value_inputs'] = []
      program.append(pg1) 

      curr_depend[countc].append(len(program)-1)

    elif semantic['operation'] == "same color":
      pg0 = {}

      if curr_depend[countc] == []:
        pg0['inputs'].append([])       
      elif curr_depend[curr_depend[countc][0]] == []:
        pg0['inputs'].append([])
      else:
        pg0['inputs'].append(curr_depend[curr_depend[countc][0]][-1])

      pg0['function'] = "equal"
      pg0['value_inputs'] = []
      pg0['value_inputs'].append("color")
      program.append(pg0)

      curr_depend[countc].append(len(program)-1)

    elif semantic['operation'] == "same material":
      pg0 = {}

      if curr_depend[countc] == []:
        pg0['inputs'].append([])       
      elif curr_depend[curr_depend[countc][0]] == []:
        pg0['inputs'].append([])
      else:
        pg0['inputs'].append(curr_depend[curr_depend[countc][0]][-1])

      pg0['function'] = "equal"
      pg0['value_inputs'] = []
      pg0['value_inputs'].append("material")
      program.append(pg0)

      curr_depend[countc].append(len(program)-1)

    elif semantic['operation'] == "same shape":
      pg0 = {}

      if curr_depend[countc] == []:
        pg0['inputs'].append([])       
      elif curr_depend[curr_depend[countc][0]] == []:
        pg0['inputs'].append([])
      else:
        pg0['inputs'].append(curr_depend[curr_depend[countc][0]][-1])

      pg0['function'] = "equal"
      pg0['value_inputs'] = []
      pg0['value_inputs'].append("shape")
      program.append(pg0)

      curr_depend[countc].append(len(program)-1)

    elif semantic['operation'] == "filter color":
      pg0 = {}

      if curr_depend[countc] == []:
        pg0['inputs'].append([])       
      elif curr_depend[curr_depend[countc][0]] == []:
        pg0['inputs'].append([])
      else:
        pg0['inputs'].append(curr_depend[curr_depend[countc][0]][-1])

      pg0['function'] = "filter_"
      pg0['value_inputs'] = []
      pg0['value_inputs'].append(semantic['argument'])
      program.append(pg0)

      curr_depend[countc].append(len(program)-1)

    elif semantic['operation'] == "filter material":
      pg0 = {}

      if curr_depend[countc] == []:
        pg0['inputs'].append([])       
      elif curr_depend[curr_depend[countc][0]] == []:
        pg0['inputs'].append([])
      else:
        pg0['inputs'].append(curr_depend[curr_depend[countc][0]][-1])

      pg0['function'] = "filter_"
      pg0['value_inputs'] = []
      pg0['value_inputs'].append(semantic['argument'])
      program.append(pg0)

      curr_depend[countc].append(len(program)-1)

    elif semantic['operation'] == "filter shape":
      pg0 = {}

      if curr_depend[countc] == []:
        pg0['inputs'].append([])       
      elif curr_depend[curr_depend[countc][0]] == []:
        pg0['inputs'].append([])
      else:
        pg0['inputs'].append(curr_depend[curr_depend[countc][0]][-1])

      pg0['function'] = "filter_"
      pg0['value_inputs'] = []
      pg0['value_inputs'].append(semantic['argument'])
      program.append(pg0)

      curr_depend[countc].append(len(program)-1)

    elif semantic['operation'] == "filter size":
      pg0 = {}

      if curr_depend[countc] == []:
        pg0['inputs'].append([])       
      elif curr_depend[curr_depend[countc][0]] == []:
        pg0['inputs'].append([])
      else:
        pg0['inputs'].append(curr_depend[curr_depend[countc][0]][-1])


      pg0['function'] = "filter_"
      pg0['value_inputs'] = []
      pg0['value_inputs'].append(semantic['argument'])
      program.append(pg0)

      curr_depend[countc].append(len(program)-1)


    elif semantic['operation'] == "select":
      pg0 = {}

      pg0['inputs'] = []
 
      if curr_depend[countc] == []:
        pg0['inputs'].append([])       
      elif curr_depend[curr_depend[countc][0]] == []:
        pg0['inputs'].append([])
      else:
        pg0['inputs'].append(curr_depend[curr_depend[countc][0]][-1])

      pg0['function'] = "filter_"
      pg0['value_inputs'] = []
      pg0['value_inputs'].append(semantic['argument'].split('(')[0])
      program.append(pg0)

      curr_depend[countc].append(len(program)-1)

    elif semantic['operation'] == "relate":
      pg0 = {}

      if curr_depend[countc] == []:
        pg0['inputs'].append([])       
      elif curr_depend[curr_depend[countc][0]] == []:
        pg0['inputs'].append([])
      else:
        pg0['inputs'].append(curr_depend[curr_depend[countc][0]][-1])


      pg0['function'] = "relate"
      pg0['value_inputs'] = []
      if semantic['argument'].split(',')[1] == "to the left of":
        pg0['value_inputs'].append("left")
      elif semantic['argument'].split(',')[1] == "to the right of":
        pg0['value_inputs'].append("right")
      elif semantic['argument'].split(',')[1] == "in front of":
        pg0['value_inputs'].append("front")
      else:
        pg0['value_inputs'].append(semantic['argument'].split(',')[1])
      program.append(pg0)

      curr_depend[countc].append(len(program)-1)

    elif semantic['operation'] == "query":
      pg0 = {}

      if curr_depend[countc] == []:
        pg0['inputs'].append([])       
      elif curr_depend[curr_depend[countc][0]] == []:
        pg0['inputs'].append([])
      else:
        pg0['inputs'].append(curr_depend[curr_depend[countc][0]][-1])


      pg0['function'] = "query"
      pg0['value_inputs'] = []
      pg0['value_inputs'].append(semantic['argument'])
      program.append(pg0)

      curr_depend[countc].append(len(program)-1)


    elif semantic['operation'] == "exist":
      pg0 = {}

      if curr_depend[countc] == []:
        pg0['inputs'].append([])       
      elif curr_depend[curr_depend[countc][0]] == []:
        pg0['inputs'].append([])
      else:
        pg0['inputs'].append(curr_depend[curr_depend[countc][0]][-1])

      pg0['function'] = "exist"
      pg0['value_inputs'] = []
      program.append(pg0)

      curr_depend[countc].append(len(program)-1)

    elif semantic['operation'] == "or":  ### ----------------------
      pg0 = {}

      if curr_depend[countc] == []:
        pg0['inputs'].append([])       
      elif curr_depend[curr_depend[countc][0]] == []:
        pg0['inputs'].append([])
      else:
        pg0['inputs'].append(curr_depend[curr_depend[countc][0]][-1])
      pg0['function'] = "union"
      pg0['value_inputs'] = []
      program.append(pg0)

      curr_depend[countc].append(len(program)-1)

    elif semantic['operation'] == "and":  ### ----------------------
      pg0 = {}

      if curr_depend[countc] == []:
        pg0['inputs'].append([])       
      elif curr_depend[curr_depend[countc][0]] == []:
        pg0['inputs'].append([])
      else:
        pg0['inputs'].append(curr_depend[curr_depend[countc][0]][-1])
      pg0['function'] = "intersect"
      pg0['value_inputs'] = []
      program.append(pg0)

      curr_depend[countc].append(len(program)-1)

    elif semantic['operation'] == "filter":
      pg0 = {}

      if curr_depend[countc] == []:
        pg0['inputs'].append([])       
      elif curr_depend[curr_depend[countc][0]] == []:
        pg0['inputs'].append([])
      else:
        pg0['inputs'].append(curr_depend[curr_depend[countc][0]][-1])

      pg0['function'] = "filter_"
      pg0['value_inputs'] = []
      pg0['value_inputs'].append(semantic['argument'])
      program.append(pg0)

      curr_depend[countc].append(len(program)-1)
  current_question['program'] = program
  vqnew_list.append(current_question)


tqnew['questions'] = tqnew_list
vqnew['questions'] = vqnew_list


with open('train_gcv0_questions.json', 'w') as f:
  json.dump(tqnew, f)

with open('val_gcv0_questions.json', 'w') as f:
  json.dump(vqnew, f)


test_1 = []
count = 0

for item in tqnew['questions']:
  count = count + 1
  test_1.append(item)
  if count > 10:
    break

test_1test = {}
test_1test['questions'] = test_1
with open('ttt_gcv0_questions.json', 'w') as f:
  json.dump(test_1test, f)

test_2 = []
count = 0

for item in vqnew['questions']:
  count = count + 1
  test_2.append(item)
  if count > 10:
    break

test_2test = {}
test_2test['questions'] = test_2
with open('vvv_gcv0_questions.json', 'w') as f:
  json.dump(test_2test, f)


  
