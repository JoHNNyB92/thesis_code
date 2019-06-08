from SPARQLWrapper import SPARQLWrapper, JSON ,DIGEST
import virtuosoWrapper.virtuosoWrapper as rdfWrapper
log_msg_ins="CRUD OPERATION(INSERT):"
class crud:
    def __init__(self):
        self.virtuosoDB = "http://localhost:8890/sparql-auth"
        self.password = "dba"
        self.username = "dba"
        self.graph = "<http://localhost:8890/DAV/anneto>"
        self.sparql = SPARQLWrapper(self.virtuosoDB)
        self.sparql.setHTTPAuth(DIGEST)
        self.sparql.setCredentials(self.username, self.password)

    def insert(self,s,o,p):
        #final_query=prefix+" INSERT DATA { GRAPH " + self.graph + " { <"+s_s+":"+s+"> <"+s_o+":"+o+"> <"+s_p+":"+p+"> } }"
        final_query="INSERT DATA { GRAPH " + self.graph + " { <"+s+"> <"+o+"> <"+p+"> } }"
        print_query="INSERT DATA { GRAPH " + self.graph + len("INSERT DATA { GRAPH " + str(self.graph))*" "+"\n{ <"+s+">\n<"+o+">\n<"+p+">\n}\n}"
        rdfWrapper.log(print_query)
        rdfWrapper.log(len(final_query)*"-")
        #print(final_query)
        #rdfWrapper.log(log_msg_ins+final_query)
        '''self.sparql.setQuery(final_query)
        self.sparql.setReturnFormat('json')
        self.sparql.method = "POST"
        res=self.sparql.query()'''


