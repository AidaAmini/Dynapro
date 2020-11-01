import json


id_index = 0
context_index = 1
question_index = 2
answer_index = 3
prediction_index = 4


class proparaColumn():
  def __init__(self, participant_name):
    self.participant_name = participant_name
    self.states = []

  def created_after_destroy(self):
    existed_ever = False
    previous_state = '-'
    for state in self.states:
      if state != '-' and previous_state == '-' and existed_ever == True:
        return False
      if state != '-':
        existed_ever = True
      previous_state = state


class proparaTable():
  def __init__(self, title, participant_names=None, columns=None):
    self.columns = columns
    self.participant_names = participant_names
    self.title = title


def read_the_rows(input_file_name):
  data = []
  input_file = open(input_file_name)
  for line in input_file:
    data_dict_row = {} 
    line_parts = line.split('\t')

    data_dict_row["id"] = line_parts[id_index]
    data_dict_row["context"] = line_parts[context_index]
    data_dict_row["question"] = line_parts[question_index]
    data_dict_row["answer"] = line_parts[answer_index]
    data_dict_row["prediction"] = line_parts[prediction_index]

    data.append(data_dict_row)

  return data

def read_data_form_json(input_json_file_name):
  with open(input_json_file_name) as f:
    data = json.load(f)
  return data
