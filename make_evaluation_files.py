import json 
import re
import argparse

def find_action(loc_before, loc_after):
  if loc_before == loc_after:
    return "NONE"
  if loc_before == '-':
    return "CREATE"
  if loc_after == '-':
    return "DESTROY"
  return "MOVE"

def read_prediction_file(prediction_file_name, int_mode=True):
  prediction_data = {}
  input_file = open(prediction_file_name)
  for line in input_file:
    if ': ' in line:
      line = line.strip().replace(".",'').replace('\"', '').replace('\'', '').replace(",", "")
      line_parts = line.split(': ')
      if int_mode == True:
        prediction_data[int(line_parts[0])] = line_parts[1].replace('.','')
      else:
        prediction_data[line_parts[0]] = line_parts[1]
  return prediction_data

def rearrange_table(table):
  index_mapping = {"id":0, "sentence_id":1, "participants":2, "action":3, "location_before":4, "location_after":5}
  participants_list = []
  for row in table:
    if row[index_mapping["participants"]] not in participants_list:
      participants_list.append(row[index_mapping["participants"]])
  
  rearranged_table = []
  for i in range(len(participants_list)-1, -1, -1):
    for j in range(len(table)):
      # print table[j][index_mapping["sentence_id"]]
      if table[j][index_mapping["participants"]] == participants_list[i]:
        rearranged_table.append(table[j])
        # print rearranged_table[len(rearranged_table)-1][index_mapping["sentence_id"]]
  return rearranged_table

###############################################################################################

def write_propara_eval_file_global_model_with_actions_simplified_rules(cat, prediction_file_name, before_state_file_name, action_file_name):
  
  int_to_action_dict = {"00":"NONE" ,'0':"NONE", '1':"MOVE", '2':"CREATE", '3':"DESTROY"}
  final_data = []
  data = []
  prediction_data = read_prediction_file(prediction_file_name, int_mode=False)
  prediction_data_before_state = read_prediction_file(before_state_file_name, int_mode=False)
  action_predicted_data = read_prediction_file(action_file_name, int_mode=False)
  with open('emnlp18_data/grids.v1.'+cat+'.json') as f:
      for line in f:
          data.append(json.loads(line))
      inconsistents_count = [0 for x in range(5)]
      for element in data:
        sentence_list = element["sentence_texts"]
        participants_list = element["participants"]
        table_data = []
        location_before = [None for x in range(len(participants_list))]
        for i in range(len(sentence_list)):
          for j in range(len(participants_list)):
            new_row = []
            new_row.append(element['para_id']) #column 1 should be the id of the process description
            new_row.append(str(i+1)) #column 2 should be the index of the sentence that we are considering
            new_row.append(participants_list[j]) #column 3 should be the name of the participant 
            index = str(int(element["para_id"])*100 + j) + '+' + str(i)
            if index + '+1' in action_predicted_data and not index.endswith('+0'):
              index = index + '+1'
            else:
              index = index + '+0'
            if index in action_predicted_data:
              action = int_to_action_dict[action_predicted_data[index]]
            else:
              action = "NONE"

            if location_before[j] == None:
              if index in prediction_data_before_state:
                location_before[j] = prediction_data_before_state[index]
              else:
                location_before[j] = 'none'
            if location_before[j] == "empty" or location_before[j] == '':
              location_before[j] = '?'
          
            inconsistent_flag = False
            inconsistents_count[0] += 1
            if action == "NONE":
                location_after = location_before[j]  
            
            #incostistent if the location before is not 'none' and location after is 'none'
            elif action == "CREATE":
              if (not(location_before[j] == 'unk' or location_before[j] == '-') or prediction_data[index] == 'unk' or prediction_data[index] == "-"):
                inconsistent_flag = True
                inconsistents_count[1] += 1
              else:
                location_after = prediction_data[index]
            
            
            #incostistent if the location after and location before are either 'none' or not equal 
            elif action == "MOVE":
              if prediction_data[index] == location_before[j] or (prediction_data[index] == 'unk' or prediction_data[index] == '-') or (location_before[j] == 'unk' or location_before[j] == '-'):
                inconsistent_flag = True
                inconsistents_count[2] += 1
              else:
                location_after = prediction_data[index]

            #incostistent if the location after is not "none" and location before is "none"
            elif action == "DESTROY":
              if location_before[j] == 'unk' or location_before[j] == '-' or (prediction_data[index] != 'unk' and prediction_data[index] != '-') :
                inconsistent_flag = True
                inconsistents_count[3] += 1
              else:
                location_after = '-'
            
            if location_before[j] == "none":
              location_before[j] = '?'
            if location_before[j] == "unk":
              location_before[j] = '-'
            if location_before[j] == '' or location_before[j] == "empty":
              if action == "NONE":
                location_before[j] = location_after
              else:
                location_before[j] = '-'
            
            #in case of incostistency we should look at the state and determinestically get the action based on that.
            if inconsistent_flag == True:
              location_after = prediction_data[index]
              if location_after == 'none' or location_after == "empty":
                location_after = '?'
              elif location_after == "unk" :
                location_after = '-'
              elif (location_after == 'empty' or location_after == '') and location_before[j] != None:
                location_after = location_before[j]
              action = find_action(location_before[j], location_after)

            if location_after == 'none' or location_after == "empty" or location_after == '':
              location_after = '?'
            elif location_after == "unk" :
              location_after = '-'

            new_row.append(action) # column 4 is the action based on location_before[j] and location_after
            new_row.append(location_before[j]) # column 5 is the location before the event
            new_row.append(location_after) # column 6 is the location that is predicted for after the event
            table_data.append(new_row)
            location_before[j] = location_after
        refined_table = rearrange_table(table_data)
        for row in refined_table:
          final_data.append(row)
  output_to_sort = []
  count = 0
  for row in final_data:
    output_to_sort.append((int(row[0]) * 10000 + count, row ))
  sorted_final_data = sorted(output_to_sort, key=lambda tup: tup[0])
  print(inconsistents_count)
  print("creating file " + "../propara_eval/data/"+cat+"/propara_format_predictions_action.tsv")
  output_file = open("../propara_eval/data/"+cat+"/propara_format_predictions_action.tsv", 'w')
  for row in sorted_final_data:
    output_file.write('\t'.join(list(row)[1]) + '\n')

################################################################################################
def write_propara_eval_file_global_model_with_actions_simplified_rules_naacl(cat, prediction_file_name, before_state_file_name, action_file_name):
  
  int_to_action_dict = {"00":"NONE" ,'0':"NONE", '1':"MOVE", '2':"CREATE", '3':"DESTROY"}
  final_data = []
  data = []
  prediction_data = read_prediction_file(prediction_file_name, int_mode=False)
  prediction_data_before_state = read_prediction_file(before_state_file_name, int_mode=False)
  action_predicted_data = read_prediction_file(action_file_name, int_mode=False)
  with open('emnlp18_data/grids.v1.'+cat+'.json') as f:
      for line in f:
          data.append(json.loads(line))
      for element in data:
        sentence_list = element["sentence_texts"]
        participants_list = element["participants"]

        naacl_par_list = []
        naacl_par_indexes = []
        for pp in range(len(participants_list)):
          participant = participants_list[pp]
          if ';' not in participant:
            naacl_par_list.append(participant)
            naacl_par_indexes.append(pp)
          else:
            par_list = participant.split(';')
            for par in par_list:
              naacl_par_list.append(par.strip())
              naacl_par_indexes.append(pp)
        participants_list = naacl_par_list
        table_data = []
        location_before = [None for x in range(len(participants_list))]
        for i in range(len(sentence_list)):
          for j in range(len(participants_list)):
            new_row = []
            new_row.append(element['para_id']) #column 1 should be the id of the process description
            new_row.append(str(i+1)) #column 2 should be the index of the sentence that we are considering
            new_row.append(participants_list[j]) #column 3 should be the name of the participant 
            index = str(int(element["para_id"])*100 + j) + '+' + str(i)
            if index + '+1' in action_predicted_data and not index.endswith('+0'):
              index = index + '+1'
            else:
              index = index + '+0'

            if index in action_predicted_data:
              action = int_to_action_dict[action_predicted_data[index]]
            else:
              action = "NONE"

            location_before[j] = prediction_data_before_state[index]
            location_after = prediction_data[index]
            
            if location_before[j] == "none":
              location_before[j] = '?'
            if location_before[j] == "unk":
              location_before[j] = '-'
            if location_after == 'none' or location_after == "empty" or location_after == '':
              location_after = '?'
            elif location_after == "unk" :
              location_after = '-'

            new_row.append(action) # column 4 is the action based on location_before[j] and location_after
            new_row.append(location_before[j]) # column 5 is the location before the event
            new_row.append(location_after) # column 6 is the location that is predicted for after the event
            table_data.append(new_row)
        refined_table = rearrange_table(table_data)
        for row in refined_table:
          final_data.append(row)
  output_to_sort = []
  count = 0
  for row in final_data:
    output_to_sort.append((int(row[0]) * 10000 + count, row ))
  sorted_final_data = sorted(output_to_sort, key=lambda tup: tup[0])
  print("creating file " + "../propara_eval/data/"+cat+"/propara_format_predictions_action.tsv")
  output_file = open("../propara_eval/data/"+cat+"/propara_format_predictions_action.tsv", 'w')
  for row in sorted_final_data:
    output_file.write('\t'.join(list(row)[1]) + '\n')

################################################################################################
def write_propara_eval_file_global_model(cat, prediction_file_name, before_prediction_file):
  final_data = []
  data = []
  prediction_data = read_prediction_file(prediction_file_name, int_mode=False)
  prediction_data_before_state = read_prediction_file(before_prediction_file, int_mode=False)
  with open('emnlp18_data/grids.v1.'+cat+'.json') as f:
      for line in f:
          data.append(json.loads(line))
      for element in data:
        sentence_list = element["sentence_texts"]
        participants_list = element["participants"]

        naacl_par_list = []
        naacl_par_indexes = []
        for pp in range(len(participants_list)):
          participant = participants_list[pp]
          if ';' not in participant:
            naacl_par_list.append(participant)
            naacl_par_indexes.append(pp)
          else:
            par_list = participant.split(';')
            for par in par_list:
              naacl_par_list.append(par.strip())
              naacl_par_indexes.append(pp)
        participants_list = naacl_par_list
        
        states_list = element["states"]
        table_data = []
        location_before = [None for x in range(len(participants_list))]
        for i in range(len(sentence_list)):
          for j in range(len(participants_list)):

            new_row = []
            new_row.append(element['para_id']) #column 1 should be the id of the process description
            new_row.append(str(i+1)) #column 2 should be the index of the sentence that we are considering
            new_row.append(participants_list[j]) #column 3 should be the name of the participant 
            index = str(int(element["para_id"])*100 + j) + '+' + str(i)
            
            if index + '+1' in prediction_data and not index.endswith('+0'):
              index = index + '+1'
            else:
              index = index + '+0'
            if index not in prediction_data.keys():
              location_after = 'none'
            else:
              location_after = prediction_data[index]
            
            if location_after == 'none' or location_after == "empty":
              location_after = '?'
            elif location_after == "unk" :
              location_after = '-'
            elif location_after == 'empty' and location_before[j] != None:
              location_after = location_before[j]
            elif location_after == 'empty' and location_before[j] != None:
              location_after = '-'
            elif location_after == '':
              location_after = '-'

            if location_before[j] == None:
              if index in prediction_data_before_state:
                location_before[j] = prediction_data_before_state[index]
              else:
                location_before[j]= location_after
            if location_before[j] == "none":
              location_before[j] = '?'
            if location_before[j] == "unk":
              location_before[j] = '-'
            if location_before[j] == '' or location_before[j] == "empty":
              location_before[j] = '-'
              # location_before[j] = location_after

            new_row.append(find_action(location_before[j], location_after)) # column 4 is the action based on location_before[j] and location_after
            new_row.append(location_before[j]) # column 5 is the location before the event
            new_row.append(location_after) # column 6 is the location that is predicted for after the event
            table_data.append(new_row)
            location_before[j] = location_after
        refined_table = rearrange_table(table_data)
        for row in refined_table:
          final_data.append(row)
  output_to_sort = []
  count = 0
  for row in final_data:
    output_to_sort.append((int(row[0]) * 10000 + count, row ))
  sorted_final_data = sorted(output_to_sort, key=lambda tup: tup[0])
  print("creating file " + "../propara_eval/data/"+cat+"/propara_format_predictions.tsv")
  output_file = open("../propara_eval/data/"+cat+"/propara_format_predictions.tsv", 'w')
  for row in sorted_final_data:
    output_file.write('\t'.join(list(row)[1]) + '\n')
##############################################################################################################

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction_file", default="predictions.json", type=str, help="The path to file containing prediction of current states.")
    parser.add_argument("--before_state_prediction_file", default="b_predictions.json", type=str, help="The path to file containing prediction of states before the current event.")
    parser.add_argument("--action_prediction_file", default="action_predictions.json", type=str, help="The path to file containing prediction actions that the participant took at every step.")
    parser.add_argument("--result_file", default="propara_format_predictions.tsv", type=str, help="The path to file that contains the resulted tables.")
    parser.add_argument("--data_devision", default="test", type=str, help="choose between the division of data if it is train, dev or test.")
    parser.add_argument("--use_actions", action='store_true', help="Whether to use the acion prediction file or not.")
    args = parser.parse_args()

    if args.use_actions:  
      write_propara_eval_file_global_model_with_actions_simplified_rules(args.data_devision, args.prediction_file, args.before_state_prediction_file, args.action_prediction_file)
    else:
      write_propara_eval_file_global_model(args.data_devision, args.prediction_file, args.before_state_prediction_file)


################################################################################################

if __name__ == "__main__":
    main()
