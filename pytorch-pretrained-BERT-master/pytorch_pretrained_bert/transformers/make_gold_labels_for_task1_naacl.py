def make_single_table_by_annotations(annotations_list, participant_list, max_sid):
  res_table = [[''for x in range(len(participant_list))] for x in range(max_sid)]
  last_sid_per_participant = [-1 for x in range(len(participant_list))]
  
  for annotation in annotations_list:
    annotation_parts = annotations.split('\t')
    participants = annotation_parts[3]

    #just a sanity check
    if ';' in participants and '; ' not in participants:
      import pdb; pdb.set_trace()
    
    annotaiton_par_list = participants.split('; ')    # There might be more thab
    current_sid = int(annotation_parts[1])
    for i in range(len(annotaiton_par_list)):
      if 


def make_tables_by_annotations(annotation_sets)


def main():
  paraIds = sys.argv[1]
  output_gold_file_name = sys.argv[2]

  output_gold_file = open(output_gold_file_name, 'w')
  input_labels_file_name = ""
  input_label_file = open(input_labels_file_name)
  


if __name__ == "__main__":
    main()

