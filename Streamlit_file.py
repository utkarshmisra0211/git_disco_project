import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from itertools import zip_longest
import networkx as nx
import streamlit as st


def load_dataset(dataset_num):
    dataset_num = st.slider("Choose a dataset", 1, 200, 1)
    file_path = f"Datasets\Sheet{dataset_num}.csv"
    df = pd.read_csv(file_path)
    return df
try:

    def process_dataset(df):
        df.loc[df['CATEGORY'] == 'X1', 'CATEGORY'] = 0.5
        df.loc[df['CATEGORY'] == 'X2', 'CATEGORY'] = 1
        df.loc[df['CATEGORY'] == 'X3', 'CATEGORY'] = 1.5

        random_ranks = np.random.permutation(np.arange(1, df.shape[0]+1))
        df['Rank'] = random_ranks
        df.sort_values('Rank', inplace=True)

        return df

    def create_preference_dict(df):
        proff_dict = {}
        for i in range(df.shape[0]):
            proff_list = df.iloc[i].tolist()
            proff_dict[proff_list[0]] = proff_list[1:]

        return proff_dict

    def create_allot_dict(FDC, HDC, FDE, HDE):
        allot_dict = {key: [] for key in set(FDC) | set(HDC) | set(FDE) | set(HDE)}
        return allot_dict

    def assign_course(prof_name, course_name, allot_dict):
        for i in allot_dict:
            if i == course_name:
                allot_dict[i].append(prof_name)

    def check_total(x):
        total = 0
        for i in x:
            total += x[i]
        return total == 0

    def preference_priority(x, FDC, HDC, FDE, HDE, proff_dict, allot_dict, only_once):
        if x == FDC:
            y = 'FDC'
        elif x == HDC:
            y = 'HDC'
        elif x == FDE:
            y = 'FDE'
        elif x == HDE:
            y = 'HDE'
        
        for i in range(2, 12):
            for name in proff_dict:
                if proff_dict[name][i].startswith(y):
                    course_name = proff_dict[name][i]
                    if course_name in x:
                        if x[course_name] in [0.5, 1] and proff_dict[name][1] in [0.5, 1.5]:
                            if proff_dict[name][1] in [0.5, 1.5] and course_name not in only_once:
                                assign_course(name, course_name, allot_dict)
                                x[course_name] -= 0.5
                                proff_dict[name][1] -= 0.5
                            elif proff_dict[name][1] == 1.5 and course_name in only_once:
                                assign_course(name, course_name, allot_dict)
                                x[course_name] -= 1
                                proff_dict[name][1] -= 1
                        elif x[course_name] == 1 and proff_dict[name][1] in [1, 1.5]:
                            assign_course(name, course_name, allot_dict)
                            x[course_name] = 0
                            proff_dict[name][1] -= 1
            
            if check_total(x):
                break

    def main():
        st.title("Course Assignment App")
        
        a = 1
        df = load_dataset(a)
        df = process_dataset(df)

        FDC = {'FDC1': 1, 'FDC2': 1, 'FDC3': 1, 'FDC4': 1, 'FDC5': 1, 'FDC6': 1, 'FDC7': 1, 'FDC8': 1, 'FDC9': 1, 'FDC10': 1, 'FDC11': 1}
        HDC = {'HDC1': 1, 'HDC2': 1, 'HDC3': 1, 'HDC4': 1}
        FDE = {'FDE1': 1, 'FDE2': 1, 'FDE3': 1, 'FDE4': 1, 'FDE5': 1, 'FDE6': 1}
        HDE = {'HDE1': 1, 'HDE2': 1, 'HDE3': 1, 'HDE4': 1}

        allot_dict = create_allot_dict(FDC, HDC, FDE, HDE)
        proff_dict = create_preference_dict(df)

        only_once = []
        for key in allot_dict:
            occurrences = df.values.flatten().tolist().count(key)
            if occurrences == 1:
                only_once.append(key)

        print("Allot Dict:", allot_dict)
        print("Proff Dict:", proff_dict)
        print("Only Once:", only_once)
        for key in proff_dict:
            for once in only_once:
                if once in proff_dict[key] and proff_dict[key][1] in [1,1.5]:
                    assign_course(key,once,allot_dict)
                    proff_dict[key][1]-=1
                    if once.startswith("FDC"):
                        FDC[once]-=1
                    elif once.startswith("HDC"):
                        HDC[once]-=1

        preference_priority(FDC, FDC, HDC, FDE, HDE, proff_dict, allot_dict, only_once)
        preference_priority(HDC, FDC, HDC, FDE, HDE, proff_dict, allot_dict, only_once)
        preference_priority(FDE, FDC, HDC, FDE, HDE, proff_dict, allot_dict, only_once)
        preference_priority(HDE, FDC, HDC, FDE, HDE, proff_dict, allot_dict, only_once)

        # print("Allot Dict After Preference Priority:", allot_dict)

        for key in FDE:
            if FDE[key] == 0.5:
                FDE[key] += 0.5
                proff = allot_dict[key][0]
                allot_dict[key] = []
                proff_dict[proff][1] += 0.5

        for key in HDE:
            if HDE[key] == 0.5:
                HDE[key] += 0.5
                proff = allot_dict[key][0]
                allot_dict[key] = []
                proff_dict[proff][1] += 0.5
        # print("Allot Dict After FDE and HDE Assignment:", allot_dict)

                
        for i in range(len(allot_dict)):
            if len(list(allot_dict.items())[i][1])>1:
                max=1
                break
            else:
                max=0
        if max==1:
            df = pd.DataFrame(allot_dict.items(), columns=['Courses', 'Professors'])
            df[['Professor1', 'Professor2']] = pd.DataFrame(df['Professors'].tolist(),index=df.index)
            df = df.drop(columns='Professors')
        
            def custom_sort(course):
                letters = course[:3]
                number = int(course[3:])
                category_order = {'FDC': 1, 'HDC': 2, 'FDE': 3, 'HDE': 4}
                return (category_order[letters], number)

            df['sort_key'] = df['Courses'].apply(custom_sort)
            df = df.sort_values('sort_key').drop(columns='sort_key')
            df = df.reset_index(drop=True)
        else:
            df = pd.DataFrame(allot_dict.items(), columns=['Courses', 'Professors'])
            df['Professor1'] = pd.DataFrame(df['Professors'].tolist(),index=df.index)
            df = df.drop(columns='Professors')
            
            def custom_sort(course):
                letters = course[:3]
                number = int(course[3:])
                category_order = {'FDC': 1, 'HDC': 2, 'FDE': 3, 'HDE': 4}
                return (category_order[letters], number)

            df['sort_key'] = df['Courses'].apply(custom_sort)
            df = df.sort_values('sort_key').drop(columns='sort_key')
            df = df.reset_index(drop=True) 


        
        # df_result = pd.DataFrame(allot_dict.items(), columns=['Courses', 'Professors'])
        # df_result[['Professor1', 'Professor2']] = pd.DataFrame(df_result['Professors'].tolist(), index=df_result.index)
        # df_result = df_result.drop(columns='Professors')


        # print("Final DataFrame:", df)

        # def red(x):
        #     if pd.isna(x):
        #         return 'font-weight: bold; color: red'
        #     else:
        #         return ''

        # styled_df = df_result.style.map(red)
        # styled_df.to_excel('styled_dataset.xlsx', index=False)

        st.dataframe(df,1000)

        # professor_counts = df[['Professor1', 'Professor2']].count(axis=1)

        # plt.figure(figsize=(10, 6))
        # plt.bar(df['Courses'], professor_counts, color='skyblue')
        # plt.xlabel('Courses')
        # plt.ylabel('Number of Professors')
        # plt.title('Number of Professors per Course')
        # plt.xticks(rotation=45, ha='right')
        # plt.tight_layout()
        # plt.show()

        # st.pyplot()
        
        if df.shape[1]==2:
            # Use zip_longest to ensure equal length, filling with None
            zipped_data = zip_longest(df['Courses'], df['Professor1'])

            # Create DataFrame from zipped data
            df = pd.DataFrame(zipped_data, columns=['Courses', 'Professor1'])

            # Drop rows where 'Courses' is None
            df = df.dropna(subset=['Courses'])

            # Continue with the rest of the code for creating and plotting the bipartite graph

            # Create a list of edges (course, professor)
            edges = []
            for _, row in df.iterrows():
                course = row['Courses']
                professor1 = row['Professor1']

                if professor1 is not None:
                    edges.append((course, professor1))

                

            # Create a bipartite graph
            B = nx.Graph()

            # Add nodes with the 'bipartite' attribute
            B.add_nodes_from(df['Courses'], bipartite=0)  # Courses
            B.add_nodes_from(df['Professor1'].dropna(), bipartite=1)  # Professors

            # Add edges to the bipartite graph
            B.add_edges_from(edges)

            # Separate nodes into two sets for bipartite layout
            courses = {node for node, bipartite in nx.get_node_attributes(B, 'bipartite').items() if bipartite == 0}
            professors = {node for node, bipartite in nx.get_node_attributes(B, 'bipartite').items() if bipartite == 1}

            # Compute bipartite layout
            pos = dict()
            pos.update((node, (1, index)) for index, node in enumerate(courses))
            pos.update((node, (2, index)) for index, node in enumerate(professors))

            # Create a new figure
            fig, ax = plt.subplots(figsize=(50, 80)) 
            # Plot the bipartite graph
            node_size = 21000
            nx.draw(B, pos, with_labels=True, font_size=40, node_size=node_size, node_color='lightblue', font_color='black',font_weight='bold',arrowsize=50, ax=ax)

            # Add a title to the plot
            ax.set_title("Bipartite Course-Professor Relationship",fontsize=70,y=0.95)
            print()
            print()
            # Display the plot
            plt.show()
        else:
            # Use zip_longest to ensure equal length, filling with None
            zipped_data = zip_longest(df['Courses'], df['Professor1'], df['Professor2'])

            # Create DataFrame from zipped data
            df = pd.DataFrame(zipped_data, columns=['Courses', 'Professor1', 'Professor2'])

            # Drop rows where 'Courses' is None
            df = df.dropna(subset=['Courses'])

            # Continue with the rest of the code for creating and plotting the bipartite graph

            # Create a list of edges (course, professor)
            edges = []
            for _, row in df.iterrows():
                course = row['Courses']
                professor1 = row['Professor1']
                professor2 = row['Professor2']

                if professor1 is not None:
                    edges.append((course, professor1))

                if professor2 is not None:
                    edges.append((course, professor2))

            # Create a bipartite graph
            B = nx.Graph()

            # Add nodes with the 'bipartite' attribute
            B.add_nodes_from(df['Courses'], bipartite=0)  # Courses
            B.add_nodes_from(df['Professor1'].dropna(), bipartite=1)  # Professors
            B.add_nodes_from(df['Professor2'].dropna(), bipartite=1)  # Professors

            # Add edges to the bipartite graph
            B.add_edges_from(edges)

            # Separate nodes into two sets for bipartite layout
            courses = {node for node, bipartite in nx.get_node_attributes(B, 'bipartite').items() if bipartite == 0}
            professors = {node for node, bipartite in nx.get_node_attributes(B, 'bipartite').items() if bipartite == 1}

            # Compute bipartite layout
            pos = dict()
            pos.update((node, (1, index)) for index, node in enumerate(courses))
            pos.update((node, (2, index)) for index, node in enumerate(professors))

            # Create a new figure
            fig, ax = plt.subplots(figsize=(5, 8)) 
            # Plot the bipartite graph
            node_size = 200
            nx.draw(B, pos, with_labels=True, font_size=4, node_size=node_size, node_color='lightblue', font_color='black',font_weight='bold', ax=ax)

            # Add a title to the plot
            ax.set_title("Bipartite Course-Professor Relationship",fontsize=7,y=0.95)
            print()
            print()
            # Display the plot
            st.pyplot(fig)

            


    if __name__ == "__main__":
        main()
except:
    st.write("Crash Test")

    



