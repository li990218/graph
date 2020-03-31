class part_table:
    def __init__(self, central, edge_num):
        self.central = central
        self.vertex_num = 1
        self.edge_num = edge_num
        self.vertex_id = []

    def add_vertex(self, vertex_id, edge_num):
        self.vertex_num += 1
        self.vertex_id.append(vertex_id)
        self.edge_num += edge_num
    
    def drop_id(self):
        del self.vertex_id[0]

    def minus_vertex(self, vertex_id, edge_num):
        self.vertex_num -= 1
        del self_vertex_id[-1]
        self.edge_num -= edge_num

