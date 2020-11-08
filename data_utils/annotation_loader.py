
class Table: 
    def __init__(self):
        self.cells = []
    
    def get_bbox(self):
        # (x1, y1), (x1, y2), (x2, y2), (x2, y1) 
        #     --> [x,y,width,height] 
        # https://cocodataset.org/#format-data
        bounds = min(self.coords[0][0], self.coords[1][0]), \
                min(self.coords[0][1], self.coords[3][1]), \
                max(self.coords[2][0], self.coords[3][0]), \
                max(self.coords[1][1], self.coords[2][1])
        bbox = bounds[0], bounds[1], \
                bounds[2] - bounds[0], bounds[3] - bounds[1]
        self.bbox = bbox
        return bbox
    
    def __str__(self):
        return "table_id="+self.id+", \n"+\
            f"table_coords={self.coords}, \n"+\
            f"table_bbox={self.get_bbox()}, \n"+\
            f"table_cells={len(self.cells)}: \n"+\
            "\n".join([str(cell) for cell in self.cells])+"\n"
    
    def __dict__(self):
        print("len", (self.cells))
        return {'id':self.id,
            'coords':self.coords,
            'bbox':self.coords,
            'cells':[cell.__dict__ for cell in self.cells]}

class Cell:
    def __init__(self):
        pass
    
    def get_bbox(self):
        # (x1, y1), (x1, y2), (x2, y2), (x2, y1) 
        #     --> [x,y,width,height] 
        # https://cocodataset.org/#format-data
        bounds = min(self.coords[0][0], self.coords[1][0]), \
                min(self.coords[0][1], self.coords[3][1]), \
                max(self.coords[2][0], self.coords[3][0]), \
                max(self.coords[1][1], self.coords[2][1])
        bbox = bounds[0], bounds[1], \
                bounds[2] - bounds[0], bounds[3] - bounds[1]
        self.bbox = bbox
        return bbox
    
    def __str__(self):
        return "\tcell_id="+self.id+", \n"+\
            f"\tcell_coords={self.coords}, \n"+\
            f"\tcell_bbox={self.bbox}, \n"+\
            f"\tcell_location={self.location} \n"
    
    """def __dict__(self):
        return {'id':self.id,
            'coords':self.coords, 
            'location':self.location}"""



def parse_tables_from_xml(xml_path):
    """Parses the ICDAR2019 xml annotation into a list of objects.

    Parameters
    ----------
    xml_path : path to the ICDAR2019 xml annotation.

    Returns
    -------
    tables : a list of :ref:`Table` objects from the xml annotation.
    """
    
    from xml.dom import minidom
    xmldoc = minidom.parse(xml_path)
    
    table_objs = []

    tablelist = xmldoc.getElementsByTagName('table')

    for table in tablelist:
        table_obj = Table()

        table_id = table.attributes['id'].value
        table_obj.id = table_id

        coords = [node for node in table.childNodes 
                    if node.nodeType == table.ELEMENT_NODE][0].attributes["points"].value
        coords = tuple([tuple(map(int, point.split(','))) 
                        for point in coords.split()])
        table_obj.coords = coords
        table_obj.get_bbox()

        for cell in table.getElementsByTagName("cell"):
            cell_obj = Cell()
            cell_id = cell.attributes['id'].value
            cell_obj.id = cell_id

            #start_col, start_row, end_col, end_row
            location_start = \
                cell.attributes["start-col"].value, \
                cell.attributes["start-row"].value, 
            try:
                location_end = \
                    cell.attributes["end-col"].value, \
                    cell.attributes["end-row"].value
            except:
                location_end = ()
            location = location_start + location_end
            location = tuple(map(int, location))
            cell_obj.location = location

            ccoords = [node for node in cell.childNodes 
                        if node.nodeType == cell.ELEMENT_NODE][0].attributes["points"].value
            ccoords = tuple([tuple(map(int, point.split(','))) 
                                for point in ccoords.split()])
            
            cell_obj.coords = ccoords
            cell_obj.get_bbox()

            table_obj.cells.append(cell_obj)
        table_objs.append(table_obj)
        
    return table_objs
