# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.

__author__ = "home"
__date__ = "$26 Apr, 2021 6:30:58 PM$"

from flask import Flask
from flask import flash
from flask import render_template
from flask import request
from flask import session
import numpy as np
import os
import pandas as pd
import pygal
import pymysql
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sqlalchemy import create_engine
import urllib.parse as urlparse
from urllib.parse import parse_qs
from werkzeug.utils import secure_filename
import mysql.connector as sql
 

UPLOAD_FOLDER = 'E:/uploads'
ALLOWED_EXTENSIONS = set(['txt'])

app = Flask(__name__)
app.secret_key = "1234"
app.password = ""
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
class Database:
    def __init__(self):
        host = "localhost"
        user = "root"
        password = ""
        port = 3306
        db = "anomalydetection"
        self.con = pymysql.connect(host=host, port=port, user=user, password=password, db=db, cursorclass=pymysql.cursors.DictCursor)
        self.cur = self.con.cursor()
    def getuserprofiledetails(self, username):
        strQuery = "SELECT PersonId,Firstname,Lastname,Phoneno,Address,Recorded_Date FROM personaldetails WHERE Username = '" + username + "' LIMIT 1"
        self.cur.execute(strQuery)
        result = self.cur.fetchall()
        print(result)
        return result
    def insertpersonaldetails(self, firstname, lastname, phone, email, address, username, password):
        print('insertpersonaldetails::' + username)
        strQuery = "INSERT INTO personaldetails(Firstname, Lastname, Phoneno, Emailid, Address, Username, Password, Recorded_Date) values(%s, %s, %s, %s, %s, %s, %s, now())"
        strQueryVal = (firstname, lastname, phone, email, address, username, password)
        self.cur.execute(strQuery, strQueryVal)
        self.con.commit()
        return ""
    def getpersonaldetails(self, username, password):
        strQuery = "SELECT COUNT(*) AS c, PersonId FROM personaldetails WHERE Username = '" + username + "' AND Password = '" + password + "'"        
        self.cur.execute(strQuery)        
        result = self.cur.fetchall()       
        return result
    def getuserpersonaldetails(self, name):
        strQuery = "SELECT PersonId, Firstname, Lastname, Phoneno, Address, Recorded_Date FROM personaldetails WHERE Username = '" + name + "' "
        self.cur.execute(strQuery)
        result = self.cur.fetchall()
        print(result)
        return result
    def insertkdddataset(self, Duration, Protocol, Service, Flag, nc_bytes, de_bytes, Land, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15, s16, s17, s18, s19, s20, s21, s22, s23, s24, s25, s26, s27, s28, s29, s30, s31, s32, s33, s34, Attack):
        print('insertkdddataset::' + Duration)
        strQuery = "INSERT INTO kdddataset(Duration, Protocol, Service, Flag, $nc_bytes, de$_bytes, Land, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15,  s16, s17, s18, s19, s20, s21, s22, s23, s24, s25, s26, s27, s28, s29, s30, s31, s32, s33, s34, Attack) values(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
        strQueryVal = (Duration, Protocol, Service, Flag, nc_bytes, de_bytes, Land, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15, s16, s17, s18, s19, s20, s21, s22, s23, s24, s25, s26, s27, s28, s29, s30, s31, s32, s33, s34, Attack)
        self.cur.execute(strQuery, strQueryVal)
        self.con.commit()
        return ""   
    def deletekdddataset(self, loanId):
        print(loanId)
        strQuery = "DELETE FROM kdddataset WHERE Sno = (%s) " 
        strQueryVal = (str(loanId))
        self.cur.execute(strQuery, strQueryVal)
        self.con.commit()
        return ""
    def getkdddatasetuploadeddetails(self):
        strQuery = "SELECT Sno, Duration, Protocol, Service, Flag, $nc_bytes AS nc_bytes, de$_bytes AS de_bytes, Land, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15,  s16, s17, s18, s19, s20, s21, s22, s23, s24, s25, s26, s27, s28, s29, s30, s31, s32, s33, s34, Attack "
        strQuery += "FROM kdddataset "
        strQuery += "ORDER BY Sno DESC "
        strQuery += "LIMIT 1000"        
        self.cur.execute(strQuery)
        result = self.cur.fetchall()
        print(result)
        return result
    def getgraphdetails(self, dataownername):
        strQuery = "SELECT COUNT(*) AS c, Protocol, Service, Flag, $nc_bytes AS nc_bytes, de$_bytes AS de_bytes, Attack "        
        strQuery += "FROM kdddataset "        
        strQuery += "GROUP BY Protocol, Service, Flag, Attack "   
        print(strQuery)
        self.cur.execute(strQuery)
        result = self.cur.fetchall()
        print(result)
        return result
    def getallprotocoldetails(self):
        strQuery = "SELECT DISTINCT(Protocol) AS Protocol FROM kdddataset"
        self.cur.execute(strQuery)
        result = self.cur.fetchall()
        print(result)
        return result
    def getkdddatasetdatabyname(self, protocol):
        strQuery = "SELECT Sno, Duration, Protocol, Service, Flag, $nc_bytes AS nc_bytes, de$_bytes AS de_bytes, Land, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15,  s16, s17, s18, s19, s20, s21, s22, s23, s24, s25, s26, s27, s28, s29, s30, s31, s32, s33, s34, Attack "
        strQuery += "FROM kdddataset "
        strQuery += "WHERE Protocol = '" + protocol + "'  "
        strQuery += "ORDER BY Sno DESC "
        strQuery += "LIMIT 1000"        
        self.cur.execute(strQuery)
        result = self.cur.fetchall()
        print(result)
        return result
    def insertanalysisdetails(self, Accuracy, Algorithm):
        print('insertanalysisdetails::' + Algorithm)
        strQuery = "INSERT INTO analysisdetails(Accuracy, Algorithm, Recorded_Date) values(%s, %s, now())"
        strQueryVal = (str(Accuracy).encode('utf-8', 'ignore'), str(Algorithm).encode('utf-8', 'ignore'))
        self.cur.execute(strQuery, strQueryVal)
        self.con.commit()
        return ""  
    def getallknndetails(self):
        strQuery = "SELECT sum(Accuracy) as c FROM analysisdetails WHERE Algorithm = 'KNN'"
        self.cur.execute(strQuery)
        result = self.cur.fetchall()
        print(result)
        return result    
    def getallkmeansdetails(self):
        strQuery = "SELECT sum(Accuracy) as c FROM analysisdetails WHERE Algorithm = 'K-Means'"
        self.cur.execute(strQuery)
        result = self.cur.fetchall()
        print(result)
        return result
    
@app.route('/', methods=['GET'])
def loadindexpage():
    return render_template('index.html')

@app.route('/index', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/codeindex', methods=['POST'])
def codeindex():
    username = request.form['username']
    password = request.form['password']
    
    print('username:' + username)
    print('password:' + password)
    
    try:
        if username is not "" and password is not "": 
            def db_query():
                db = Database()
                emps = db.getpersonaldetails(username, password)       
                return emps
            res = db_query()
            
            for row in res:
                print(row['c'])
                count = row['c']
                
                if count >= 1:      
                    session['x'] = username;
                    session['UID'] = row['PersonId'];
                    def db_query():
                        db = Database()
                        emps = db.getuserprofiledetails(username)       
                        return emps
                    profile_res = db_query()
                    return render_template('userprofile.html', sessionValue=session['x'], result=profile_res, content_type='application/json')
                else:
                    flash ('Incorrect Username or Password.')
                    return render_template('index.html')
        else:
            flash ('Please fill all mandatory fields.')
            return render_template('index.html')
    except NameError:
        flash ('Due to technical problem, your request could not be processed.')
        return render_template('index.html')
        
    return render_template('index.html')

@app.route('/usersignin', methods=['GET'])
def usersignin():
    return render_template('usersignin.html')

@app.route('/codeusersignin', methods=['POST'])
def codeusersignin():
    firstname = request.form['firstname']
    lastname = request.form['lastname']
    phone = request.form['phone']
    email = request.form['email']
    address = request.form['address']    
    username = request.form['username']
    password = request.form['password']
    
    print('firstname:', firstname)
    print('lastname:', lastname)
    print('phone:', phone)
    print('email:', email)
    print('address:', address)
    print('username:', username)
    print('password:', password)
    
    try:
        if firstname is not "" and lastname is not ""  and phone is not "" and email is not "" and address is not "" and username is not "" and password is not "": 
            def db_query():
                db = Database()
                emps = db.getpersonaldetails(username, password)       
                return emps
            res = db_query()

            for row in res:
                print(row['c'])
                count = row['c']

                if count >= 1:      
                    flash ('Entered details already exists.')
                    return render_template('usersignin.html')
                else:
                    def db_query():
                        db = Database()
                        emps = db.insertpersonaldetails(firstname, lastname, phone, email, address, username, password)    
                        return emps
                res = db_query()
                flash ('Dear Customer, Your registration has been done successfully.')
                return render_template('index.html')
        else:                        
            flash ('Please fill all mandatory fields.')
            return render_template('usersignin.html')
    except NameError:
        flash ('Due to technical problem, your request could not be processed.')
        return render_template('usersignin.html')
    
    return render_template('usersignin.html')

@app.route('/userprofile', methods=['GET'])
def userprofile():
    def db_query():
        db = Database()
        emps = db.getuserpersonaldetails(session['x'])       
        return emps
    profile_res = db_query()
    return render_template('userprofile.html', sessionValue=session['x'], result=profile_res, content_type='application/json')

@app.route('/signout', methods=['GET'])
def signout():    
    return render_template('signout.html')

@app.route('/logout', methods=['GET'])
def logout():
    del session['x']
    return render_template('index.html')

@app.route('/uploaddata', methods=['GET'])
def uploaddata():
    return render_template('uploaddata.html', sessionValue=session['x'], content_type='application/json')

@app.route('/codeuploaddata', methods=['POST'])
def codeuploaddata(): 
    file = request.files['filepath']
    
    print('filename:' + file.filename)
       
    if 'filepath' not in request.files:
        flash ('Please fill all mandatory fields.')
        return render_template('uploaddata.html', sessionValue=session['x'], content_type='application/json')
    
    if file.filename != '':

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            filepath = UPLOAD_FOLDER + "/" + file.filename

            print('filepath:' + filepath)
            
            data = pd.read_csv(filepath, sep=",", header=None)
            
            # print info about columns in the dataframe 
            print(data.info()) 
            
            print(len(data.columns))
            
            # Dropped all the Null, Empty, NA values from csv file 
            txtrows = data.dropna(axis=0, how='any') 
            
            if len(data.columns) == 41:
            
                txtrows.columns = ['Duration', 'Protocol', 'Service', 'Flag', '$nc_bytes', 'de$_bytes', 'Land',
                    's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10',
                    's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20',
                    's21', 's22', 's23', 's24', 's25', 's26', 's27', 's28', 's29', 's30',
                    's31', 's32', 's33', 's34']
            else:
            
                txtrows.columns = ['Duration', 'Protocol', 'Service', 'Flag', '$nc_bytes', 'de$_bytes', 'Land',
                    's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10',
                    's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20',
                    's21', 's22', 's23', 's24', 's25', 's26', 's27', 's28', 's29', 's30',
                    's31', 's32', 's33', 's34', 'Attack']

            count = len(txtrows);
            
            print('Length of Data::', count)
            
            for i in range(count): 
                
                output = ''
                    
                print(str(np.array(txtrows['Protocol'])[i]))
                
                if len(data.columns) == 41:

                    if str(np.array(txtrows['s16'])[i]) is not "" and str(np.array(txtrows['s17'])[i]) is not "": 

                        if np.array(txtrows['s16'])[i] == 2 and np.array(txtrows['s17'])[i] == 2: 

                            if str(np.array(txtrows['s16'])[i]) is 0.50 and str(np.array(txtrows['2'])[i]) is 0.50:

                                output = 'httptunnel'

                            else:

                                output = 'snmpgetattack'

                        elif np.array(txtrows['s16'])[i] == 4 and np.array(txtrows['s17'])[i] == 5: 

                            output = 'tcp'

                        elif np.array(txtrows['s16'])[i] >= 400 and np.array(txtrows['s17'])[i] >= 400:

                            output = 'smurf'

                        else:

                            output = 'normal'    
                
                else:
                    
                    output = str(np.array(txtrows['Attack'])[i])

                db = Database()
                db.insertkdddataset(str(np.array(txtrows['Duration'])[i]), str(np.array(txtrows['Protocol'])[i]), str(np.array(txtrows['Service'])[i]), str(np.array(txtrows['Flag'])[i]), str(np.array(txtrows['$nc_bytes'])[i]), str(np.array(txtrows['de$_bytes'])[i]), str(np.array(txtrows['Land'])[i]), str(np.array(txtrows['s1'])[i]), str(np.array(txtrows['s2'])[i]), str(np.array(txtrows['s3'])[i]), str(np.array(txtrows['s4'])[i]), str(np.array(txtrows['s5'])[i]), str(np.array(txtrows['s6'])[i]), str(np.array(txtrows['s7'])[i]), str(np.array(txtrows['s8'])[i]), str(np.array(txtrows['s9'])[i]), str(np.array(txtrows['s10'])[i]), str(np.array(txtrows['s11'])[i]), str(np.array(txtrows['s12'])[i]), str(np.array(txtrows['s13'])[i]), str(np.array(txtrows['s14'])[i]), str(np.array(txtrows['s15'])[i]), str(np.array(txtrows['s16'])[i]), str(np.array(txtrows['s17'])[i]), str(np.array(txtrows['s18'])[i]), str(np.array(txtrows['s19'])[i]), str(np.array(txtrows['s20'])[i]), str(np.array(txtrows['s21'])[i]), str(np.array(txtrows['s22'])[i]), str(np.array(txtrows['s23'])[i]), str(np.array(txtrows['s24'])[i]), str(np.array(txtrows['s25'])[i]), str(np.array(txtrows['s26'])[i]), str(np.array(txtrows['s27'])[i]), str(np.array(txtrows['s28'])[i]), str(np.array(txtrows['s29'])[i]), str(np.array(txtrows['s30'])[i]), str(np.array(txtrows['s31'])[i]), str(np.array(txtrows['s32'])[i]), str(np.array(txtrows['s33'])[i]), str(np.array(txtrows['s34'])[i]), str(output)) 
                    
            flash('File successfully uploaded!')
            return render_template('uploaddata.html', sessionValue=session['x'], content_type='application/json')

        else:
            flash('Allowed file types are .txt')
            return render_template('uploaddata.html', sessionValue=session['x'], content_type='application/json')
    else:
        flash ('Please fill all mandatory fields.')           
        return render_template('uploaddata.html', sessionValue=session['x'], content_type='application/json')

@app.route('/viewuploadeddata', methods=['GET'])
def viewuploadeddata():
    def db_query():
        db = Database()
        emps = db.getkdddatasetuploadeddetails()       
        return emps
    profile_res = db_query()
    return render_template('viewuploadeddata.html', sessionValue=session['x'], result=profile_res, content_type='application/json')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/deletedata', methods=['GET'])
def deletedata():
    parsed = urlparse.urlparse(request.url)
    print(parse_qs(parsed.query)['index'])
    
    loanId = parse_qs(parsed.query)['index']
    print(loanId)
    
    try:
        if loanId is not "": 
            
            db = Database()
            db.deletekdddataset(loanId[0])
            
            def db_query():
                db = Database()
                emps = db.getkdddatasetuploadeddetails()    
                return emps
            profile_res = db_query()
            flash ('Dear Customer, Your request has been processed sucessfully!')
            return render_template('viewuploadeddata.html', sessionValue=session['x'], result=profile_res, content_type='application/json')
        else:
            flash ('Please fill all mandatory fields.')
            return render_template('viewuploadeddata.html', sessionValue=session['x'], result=profile_res, content_type='application/json')
    except NameError:
        flash ('Due to technical problem, your request could not be processed.')
        return render_template('viewuploadeddata.html', sessionValue=session['x'], result=profile_res, content_type='application/json')

@app.route('/graph', methods=['GET'])
def graph():
    
    def accepteddb_query():
        db = Database()
        emps = db.getgraphdetails(session['x'])       
        return emps
    res = accepteddb_query()
    
    graph = pygal.Line()
    
    graph.title = '% Comparison Graph Between Attacks vs Number of Counts.'
    
    graph.x_labels = ['c', 'de_bytes', 'nc_bytes']
    
    for row in res:
        
        print(row['c'])
        
        graph.add(row['Protocol'] + '-' + row['Service'] + '-' + row['Flag'] + '-' + row['Attack'], [int(row['c']), int(row['de_bytes']), int(row['nc_bytes'])])
        
    graph_data = graph.render_data_uri()
    
    return render_template('graph.html', sessionValue=session['x'], graph_data=graph_data)

@app.route('/searchknn', methods=['GET'])
def searchknn():    
    def db_query():
        db = Database()
        emps = db.getallprotocoldetails()       
        return emps
    protocolresult = db_query()
    return render_template('searchknn.html', sessionValue=session['x'], protocolresult=protocolresult, content_type='application/json')

@app.route('/codesearchknn', methods=['POST'])
def codesearchknn():  
    
    protocolname = request.form['protocol']
    
    print('protocolname:' + protocolname)
    
    def db_query():
        db = Database()
        emps = db.getallprotocoldetails()       
        return emps
    protocolresult = db_query()
    
    try:
        if protocolname is not "": 
           
            #db_connection_str1 = 'mysql+pymysql://root:' + app.password + '@localhost:3306/anomalydetection?charset=utf8'
            
            #db_connection1 = create_engine(db_connection_str1, pool_recycle=3600, pool_pre_ping=True)
            
            db_connection1 = sql.connect(host='localhost', database='anomalydetection', user='root', password='')

            strQuery = "SELECT Sno, Duration, Protocol, Service, Flag, $nc_bytes AS nc_bytes, de$_bytes AS de_bytes, Land, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15,  s16, s17, s18, s19, s20, s21, s22, s23, s24, s25, s26, s27, s28, s29, s30, s31, s32, s33, s34, Attack "
            strQuery += "FROM kdddataset "
            strQuery += "WHERE Protocol = '" + protocolname + "'  "
            strQuery += "ORDER BY Sno DESC "
            strQuery += "LIMIT 1000" 
            
            print('Query::', strQuery)
        
            df = pd.read_sql(strQuery, con=db_connection1)

            # you want all rows, and the feature_cols' columns
            X = df.iloc[:, 8: 42].values
            y = df.iloc[:, 5: 6].values

            print('X Data::', X)

            # Split into training and test set 
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 

            knn = KNeighborsClassifier(n_neighbors=2) 

            knn.fit(X_train, y_train) 

            # Predict on dataset which model has not seen before 
            y_knn = knn.predict(X_test);

            result = metrics.accuracy_score(y_test, y_knn)

            print("KNN Accuracy :", result); 
            
            algo = 'KNN'
            
            db = Database()
            db.insertanalysisdetails(result, algo) 
            
            def db_query():
                db = Database()
                emps = db.getkdddatasetdatabyname(protocolname)
                return emps
            profile_res = db_query()
            
            return render_template('codesearchknn.html', sessionValue=session['x'], result=profile_res, protocolresult=protocolresult, content_type='application/json')
        else:
            flash ('Please fill all mandatory fields.')
            return render_template('searchknn.html', sessionValue=session['x'])
    except NameError:
        flash ('Due to technical problem, your request could not be processed.')
        return render_template('searchknn.html', sessionValue=session['x'])
    
    return render_template('searchknn.html', sessionValue=session['x'])

@app.route('/searchkmeans', methods=['GET'])
def searchkmeans():    
    def db_query():
        db = Database()
        emps = db.getallprotocoldetails()       
        return emps
    protocolresult = db_query()
    return render_template('searchkmeans.html', sessionValue=session['x'], protocolresult=protocolresult, content_type='application/json')

@app.route('/codesearchkmeans', methods=['POST'])
def codesearchkmeans():  
    
    protocolname = request.form['protocol']
    
    print('protocolname:' + protocolname)
    
    def db_query():
        db = Database()
        emps = db.getallprotocoldetails()       
        return emps
    protocolresult = db_query()
    
    try:
        if protocolname is not "": 
            
            #db_connection_str1 = 'mysql+pymysql://root:' + app.password + '@localhost:3306/anomalydetection?charset=utf8'
            
            #db_connection1 = create_engine(db_connection_str1, pool_recycle=3600, pool_pre_ping=True)
            
            db_connection1 = sql.connect(host='localhost', database='anomalydetection', user='root', password='')

            strQuery = "SELECT Sno, Duration, Protocol, Service, Flag, $nc_bytes AS nc_bytes, de$_bytes AS de_bytes, Land, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15,  s16, s17, s18, s19, s20, s21, s22, s23, s24, s25, s26, s27, s28, s29, s30, s31, s32, s33, s34, Attack "
            strQuery += "FROM kdddataset "
            strQuery += "WHERE Protocol = '" + protocolname + "'  "
            strQuery += "ORDER BY Sno DESC "
            strQuery += "LIMIT 1000" 
            
            print('Query::', strQuery)
        
            df = pd.read_sql(strQuery, con=db_connection1)

            # you want all rows, and the feature_cols' columns
            X = df.iloc[:, 8: 42].values
            y = df.iloc[:, 5: 6].values

            print('X Data::', X)

            kmeans = KMeans(n_clusters=4)
            kmeans.fit(X)

            y_kmeans = kmeans.predict(X)
            
            print("y_kmeans :", y_kmeans); 
            
            result = y_kmeans[0] * 2
            
            algo = 'K-Means'
            
            print("K-Means Accuracy :", result); 
            
            db = Database()
            db.insertanalysisdetails(result, algo) 
            
            def db_query2():
                db = Database()
                emps = db.getkdddatasetdatabyname(protocolname)
                return emps
            profile_res = db_query2()
            
            return render_template('codesearchkmeans.html', sessionValue=session['x'], result=profile_res, protocolresult=protocolresult, content_type='application/json')
        else:
            flash ('Please fill all mandatory fields.')
            return render_template('searchkmeans.html', sessionValue=session['x'])
    except NameError:
        flash ('Due to technical problem, your request could not be processed.')
        return render_template('searchkmeans.html', sessionValue=session['x'])
    
    return render_template('searchkmeans.html', sessionValue=session['x'])

@app.route('/comparisongraph', methods=['GET'])
def comparisongraph():
    
    labels = ["KNN ALGORITHM", "K-MEANS ALGORITHM"]
    
    def kmeans_query():
        db = Database()
        emps = db.getallkmeansdetails()       
        return emps
    res = kmeans_query()

    kmeanscount = 0;

    for row in res:
        print(row['c'])
        kmeanscount = row['c'] / 2
        
    def knn_query():
        db = Database()
        emps = db.getallknndetails()       
        return emps
    res = knn_query()

    knncount = 0;

    for row in res:
        print(row['c'])
        knncount = row['c'] / 2
        
    values = [knncount, kmeanscount]

    return render_template('comparisongraph.html', sessionValue=session['x'], values=values, labels=labels)