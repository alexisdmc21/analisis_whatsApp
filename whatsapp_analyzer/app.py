import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
import emoji
import string
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from datetime import datetime
import base64
from io import BytesIO

st.set_page_config(
    page_title="Análisis de Chats WhatsApp - Examen Final IA",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #25D366;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .metric-card {
        background: linear-gradient(90deg, #25D366 0%, #128C7E 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .section-header {
        color: #128C7E;
        font-size: 1.5rem;
        font-weight: bold;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .exam-info {
        color: #000000
        background-color: #f0f8ff;
        padding: 1rem;
        border-left: 5px solid #25D366;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ========================= FUNCIONES DE UTILS.IPYNB =========================

def delete_tilde(texto):
    """Función para eliminar tildes y convertir mayúsculas en minúsculas"""
    list_char = ['á','é','í','ó','ú','ü']
    list_sust = ['a','e','i','o','u','u']

    i = 0
    for x in list_char:
        if x in texto:
            texto = texto.replace(x,list_sust[i])
        i += 1

    texto = texto.lower()    
    return texto

def remove_puntuation(texto):
    """Función para eliminar símbolos en un texto"""
    texto = re.sub('[%s]'% re.escape(string.punctuation), '', texto)
    return texto

def regex_word(texto):
    """Función que elimina palabras de poca relevancia"""
    word = ['\\s\\<(M|m)ultimedia\\somitido\\>', '\\somitido\\s', '\\smultimedia\\s','https?\\S*',
    '(\\<Multimedia\\s)', '\\w+\\.vcf','\\(archivo\\sadjunto\\)',
    'omitido\\>', '\\s{4}', '\\s{3}', '\\s{2}', '\\s[a-zA-Z]\\s',
    '\\svcf', '\\s(p|P)\\s(m|M)\\s', '\\s(p|P)(m|M)\\s', '\\sp\\s',
    '\\sm\\s', '\\sde\\s', '\\scon\\s', '\\sque\\s', '\\sla\\s',
    '\\slo\\s', '\\spara\\s', '\\ses\\s', '\\sdel\\s', '\\spor\\s',
    '\\sel\\s', '\\sen\\s', '\\slos\\s', '\\stu\\s', '\\ste\\s',
    '[\\w\\._]{5,30}\\+?[\\w]{0,10}@[\\w\\.\\-]{3,}\\.\\w{2,5}',
    '\\sun\\s', '\\sus\\s', 'su\\s', '\\s\\u200e', '\\u200e' '\\s\\s',
    '\\s\\s\\s', '\\s\\u200e3', '\\s\\u200e2', '\\s\\.\\.\\.\\s', '/',
    '\\s\\u200e4', '\\s\\u200e7', '\\s\\u200e8', '\\suna\\s',
    'la\\s', '\\slas\\s', '\\sse\\s', '\\sal\\s','\\sle\\s',
    '\\sbuenas\\s', '\\sbuenos\\s', '\\sdias\\s', '\\stardes\\s', '\\snoches\\s',
    '\\sesta\\s', '\\spero\\s','\\sdia\\s', '\\sbuenas\\s', '\\spuede\\s', '\\spueden\\s',
    '\\sson\\s', '\\shay\\s', '\\seste\\s', '\\scomo\\s', '\\salgun\\s', '\\salguien\\s',
    '\\stodo\\s', '\\stodos\\s', '\\snos\\s', '\\squien\\s', '\\seso\\s', '\\sdesde\\s',
    '\\sarchivo\\sadjunto\\s', 'gmailcom', '\\sdonde\\s', '\\shernan\\s', '\\slavadoras\\s',
    'gracias', '\\selimino\\smensaje\\s', '\\snnnn\\s',
    '\\sllll\\s', '\\slll/\\s', 'llll']

    regexes = [re.compile(p) for p in word]

    for regex in regexes:
            patron = re.compile(regex)
            texto = patron.sub(' ', texto)
    return texto

def delete_emoji(texto):
    """Función que elimina emojis de un texto"""
    return emoji.replace_emoji(texto, replace='')

def Nube_Words(df, fecha):
    """Función que crea una nube de palabras a partir de un dataframe"""
    text = ' '.join(df['Message'])

    # aplicando las funciones de limpieza
    text = delete_emoji(text)
    text = delete_tilde(text)
    text = remove_puntuation(text)
    text = regex_word(text)

    # nube de palabras
    wordcloud = WordCloud(stopwords=STOPWORDS, background_color='white').generate(text)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(f'{fecha}')
    
    return fig

# ========================= FUNCIONES DEL CÓDIGO USADO EN EL EXAMEN DEL SEGUNDO PARCIAL =========================

def Date_Chat(l):
    pattern = r'^\d{1,2}/\d{1,2}/\d{4},\s\d{1,2}:\d{2}\s[p|a]\.\s[m]\.\s-'
    result = re.match(pattern, l)
    if result:
        return True
    return False

def IsAuthor(l):
    pattern = '([+]\d{2} \d{3} \d{7}):'
    patterns = '^' + '|'.join(pattern)
    result = re.match(patterns, l)
    # retorna True si encuentra un numero telefonico o nombre de usuario
    if result:
        return True
    return False

def DataPoint(line):
    SplitLine = line.split(' - ') # divide la linea en el signo -
    DT = SplitLine[0] # almacena la primera parte de la linea en DT
    # Ajustar para tu formato con coma
    DateTime = DT.split(', ') # divide la linea donde haya espacios
    Date = DateTime[0] # almacena la fecha en Date 
    Format = DateTime[1].split(' ') # almacena el formato completo "p. m."
    Time = Format[0] # almacena la hora en Time
    if len(Format)>=3:
        Format = Format[1] + ' ' + Format[2]
    else:
        Format = ""

    Message = ' '.join(SplitLine[1:]) # almacena el mensaje en Message
    if IsAuthor(Message):
        authormes = Message.split(': ') # si existe el autor y Message no esta vacio, divide
        Author = authormes[0] # almacena la primera parte de Message que corresponde al autor
        Message = ' '.join(authormes[1:]) # almacena la segunda parte de Message en Message
    else:
        Author = None
    return Date, Time, Format, Author, Message

def DataFrame_Data(file_content):
    """Función adaptada para Streamlit"""
    parsedData = []
    
    # Convertir bytes a string
    content = file_content.decode('utf-8')
    lines = content.split('\n')
    
    messageBuffer = []
    Date, Time, Format, Author = None, None, None, None
    
    # Recorremos el archivo linea por linea
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # si hay una fecha en la linea y si no es vacia, añade a pasedData las variables
        if Date_Chat(line):
            if len(messageBuffer) > 0:
                parsedData.append([Date, Time, Format, Author, ' '.join(messageBuffer)])
            # limpia la lista que almacena el mensaje
            messageBuffer.clear()
            # añade a cada variable su correpondiente valor de linea
            Date, Time, Format, Author, Message = DataPoint(line)
            # añade el mensaje a la lista messageBuffer
            messageBuffer.append(Message)
        else:
            messageBuffer.append(line)
    
    # Añadir el último mensaje si existe
    if Date and len(messageBuffer) > 0:
        parsedData.append([Date, Time, Format, Author, ' '.join(messageBuffer)])
    
    # crear dataframe con los datos tratados
    df = pd.DataFrame(parsedData, columns=['Date', 'Time', 'Format', 'Author', 'Message'])
    return df

def protec_info(string):
    """Funcion que codifica la informacion sensible en los mensajes"""
    # codificar numeros de telefonos
    patron = r'\b\d{3}[-\s]?\d{7}\b|\(\d{3}\)[-.\s]?\d{3}[-.\s]?\d{4}\b'
    string = re.sub(patron,' NNNN ',string)
    # codificar links
    patron = r'\bhttps?://\S+\b'
    string = re.sub(patron,'LLLL',string)
    # codificar nombres de usuarios
    patron = r'@\w+\s?'
    string = re.sub(patron,'UUUU ',string)
    # codificar enlaces a whatsapp
    patron = r'\bwa.me/\d+\b'
    string = re.sub(patron,'NNNN',string)
    return string

def process_dataframe_complete(df):
    df = df.drop(range(0,1))
    # reseteamos los indices
    df = df.reset_index(drop=True)
    
    df = df.drop(df[df['Message'] == ''].index)
    # reseteamos los indices para los registros restantes
    df = df.reset_index(drop=True)
    
    miembros = df['Author'].unique()

    for i in enumerate(miembros):
        df['Author'] = df['Author'].replace(i[1], f'user {i[0]}')

    df['Message_Private'] = df['Message'].apply(protec_info)
    # Eliminamos la columna Message y renombramos la nueva columna
    df = df.drop('Message', axis=1)
    df = df.rename(columns={'Message_Private':'Message'})
    
    NoneValue = df[df['Author'].isnull()]
    # eliminamos los registros vacios
    df = df.drop(NoneValue.index)
    
    # unimos los valores de Format con Time en la columna Time
    df['Time'] = df.Time.str.cat(df.Format, sep=' ')
    # convertir valores de Time en objetos DateTime
    df['Time'] = pd.to_datetime(df['Time'])
    # convertir valores de Time en str
    df['Time'] = df['Time'].astype(str)
    # Extraer solo la hora de la columna Time
    df['Time'] = df['Time'].str.extract('(..:..:..)', expand=True)
    # eliminar columna Format
    df = df.drop(['Format'],axis=1)
    
    # diccionario con el valor y nombre de los dias
    week = {
        6:'Domingo',
        0:'Lunes',
        1:'Martes',
        2:'Miercoles',
        3:'Jueves',
        4:'Viernes',
        5:'Sabado'
    }

    # Diccionario con el valor y nombre de los meses
    month = {
        1:'Ene',
        2:'Feb',
        3:'Mar',
        4:'Abr',
        5:'May',
        6:'Jun',
        7:'Jul',
        8:'Ago',
        9:'Sept',
        10:'Obt',
        11:'Nov',
        12:'Dic'
    }
    
    # dar formato de fecha a los dias
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    # nueva columna con los nombres de los dias
    df['Day'] = df['Date'].dt.weekday.map(week)
    # nueva columna con el numero del dia
    df['Num_Day'] = df['Date'].dt.day
    # nueva columna con el año
    df['Year'] = df['Date'].dt.year
    # nueva columna con el numero de mes
    df['Num_Month'] = df['Date'].dt.month
    # nueva columna con el nombre de los meses
    df['Month'] = df['Date'].dt.month.map(month)
    # organizar las columnas
    df = df[['Date', 'Day', 'Num_Day','Num_Month', 'Month', 'Year', 'Time', 'Author', 'Message']]
    # cambiar el tipo de dato de la columna Day
    df['Day'] = df['Day'].astype('category')
    
    # PÁGINA 11 - EXACTO del PDF
    # df['fecha'] = pd.to_datetime(df['Date'])
    df['fecha'] = pd.to_datetime(df['Date'])
    # nueva columna con el formato indicado
    df['fecha'] = df['fecha'].dt.strftime('%d/%m/%Y')
    # eliminar columna Date
    df = df.drop(['Date'],axis=1)
    # renombrar columna fecha como Date
    df = df.rename(columns={'fecha':'Date'})
    # ordenar columnas del dataframe
    df = df[['Date','Day', 'Num_Day', 'Month', 'Num_Month' , 'Year', 'Time', 'Author', 'Message']]
    
    df['Letters'] = df['Message'].apply(lambda s:len(s))
    # df['Words'] = df['Message'].apply(lambda s:len(s.split(' ')))
    df['Words'] = df['Message'].apply(lambda s:len(s.split(' ')))
    # contar el numero de links por mensaje y guardarlo en la nueva columna
    df['URL_count'] = df.Message.apply(lambda x:x.count('LLLL'))
    
    return df

    df['Time'] = df.Time.str.cat(df.Format, sep=' ')
    # convertir valores de Time en objetos DateTime
    try:
        df['Time'] = pd.to_datetime(df['Time'])
        # convertir valores de Time en str
        df['Time'] = df['Time'].astype(str)
        # Extraer solo la hora de la columna Time
        df['Time'] = df['Time'].str.extract('(..:..:..)', expand=True)
    except:
        pass
    # eliminar columna Format
    df = df.drop(['Format'],axis=1)

    week = {
        6:'Domingo',
        0:'Lunes',
        1:'Martes',
        2:'Miercoles',
        3:'Jueves',
        4:'Viernes',
        5:'Sabado'
    }

    # Diccionario con el valor y nombre de los meses
    month = {
        1:'Ene',
        2:'Feb',
        3:'Mar',
        4:'Abr',
        5:'May',
        6:'Jun',
        7:'Jul',
        8:'Ago',
        9:'Sept',
        10:'Oct',
        11:'Nov',
        12:'Dic'
    }
    
    try:
        # dar formato de fecha a los dias
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
        # nueva columna con los nombres de los dias
        df['Day'] = df['Date'].dt.weekday.map(week)
        # nueva columna con el numero del dia
        df['Num_Day'] = df['Date'].dt.day
        # nueva columna con el año
        df['Year'] = df['Date'].dt.year
        # nueva columna con el numero de mes
        df['Num_Month'] = df['Date'].dt.month
        # nueva columna con el nombre de los meses
        df['Month'] = df['Date'].dt.month.map(month)
        # organizar las columnas
        df = df[['Date', 'Day', 'Num_Day','Num_Month', 'Month', 'Year', 'Time', 'Author', 'Message']]
        # cambiar el tipo de dato de la columna Day
        df['Day'] = df['Day'].astype('category')

        df['fecha'] = pd.to_datetime(df['Date'])
        # nueva columna con el formato indicado
        df['fecha'] = df['fecha'].dt.strftime('%d/%m/%Y')
        # eliminar columna Date
        df = df.drop(['Date'],axis=1)
        # renombrar columna fecha como Date
        df = df.rename(columns={'fecha':'Date'})
        # ordenar columnas del dataframe
        df = df[['Date','Day', 'Num_Day', 'Month', 'Num_Month', 'Year', 'Time', 'Author', 'Message']]
    except Exception as e:
        st.error(f"Error procesando fechas: {e}")
        return None
    
    df['Letters'] = df['Message'].apply(lambda s:len(s))
    df['Words'] = df['Message'].apply(lambda s:len(s.split(' ')))
    # contar el numero de links por mensaje y guardarlo en la nueva columna
    df['URL_count'] = df.Message.apply(lambda x:x.count('LLLL'))
    
    return df

# ========================= ANÁLISIS =========================

def generate_member_stats(df):
    # dataframe sin multimedia
    df_noMedia = df[df['Message'] != '<Multimedia omitido>']
    
    # listamos los miembros del grupo
    member = df.Author.unique()
    # creamos un Dataframe para guardar los valores
    stat_df = pd.DataFrame(columns=['Author', 'N_Message', 'N_Multimedia', 'Prom_Words'])
    # creamos una lista donde gurdamos los valores que vamos a pasar al DataFrame
    list_val = []
    
    # iteramos sobre cada miembro para mostrar su respectiva informacion
    for i in enumerate(member):
        # filtramos los mensajes de cada miembro
        member_data = df[df['Author'] == i[1]]
        # filtramos los mensajes de cada miembro que no sean multimedia
        member_noMedia = df_noMedia[df_noMedia['Author'] == i[1]]
        # cantidad de mensajes multimedia
        media = sum(member_data['Message'] == '<Multimedia omitido>')
        # promedio de palabras por mensaje
        if member_noMedia.shape[0] == 0:
            word_per_message = 0
        else:
            word_per_message = np.sum(member_noMedia['Words'])/member_noMedia.shape[0]
            word_per_message = ('%.2f' % round(word_per_message, 2))
        
        list_val = [
            {'Author':i[1],
             'N_Message':member_data.shape[0],
             'N_Multimedia':media,
             'Prom_Words':word_per_message}
        ]
        stat_df = pd.concat([stat_df, pd.DataFrame(list_val)], ignore_index=True)
    
    return stat_df

# ========================= INTERFAZ STREAMLIT =========================

def main():
    # Header principal
    st.markdown('<h1 class="main-header">💬 Análisis de Chats WhatsApp - Examen IA</h1>', unsafe_allow_html=True)
    
    # Información del examen
    st.markdown("""
    <div class="exam-info">
        <h4>📚 Examen Final - Inteligencia Artificial</h4>
        <p><strong>Estudiante:</strong> Alexis Damian Morales Cuásquer | <strong>NRC:</strong> 23388 | <strong>Cédula:</strong> 1750522532</p>
        <p>Esta aplicación muestra un analisis de un chat de whatsapp</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 📁 Cargar Archivo de Chat")
        uploaded_file = st.file_uploader(
            "Selecciona tu archivo de chat de WhatsApp (.txt)",
            type=['txt'],
            help="Exporta tu chat desde WhatsApp: Menú → Exportar chat → Sin multimedia"
        )
        
        if uploaded_file is not None:
            st.success("✅ Archivo cargado exitosamente!")
            
        st.markdown("### ℹ️ Proyecto realizado en base a:")
        st.markdown("""
        **Código usado en el examen del segundo parcial:**
        `Examen_2P_Practico_Morales_Alexis.ipynb`
        `utils.ipynb`
        """)
    
    if uploaded_file is not None:
        try:
            # Procesamiento usando las funciones EXACTAS del Colab
            with st.spinner("📊 Procesando archivo..."):
                # Paso 1: Crear DataFrame inicial (función del examen)
                df = DataFrame_Data(uploaded_file.read())
                
                if df.empty:
                    st.error("❌ No se pudo procesar el archivo. Verifica que sea un chat válido de WhatsApp.")
                    return
                
                # Paso 2: Procesar DataFrame completo (todos los pasos del Colab)
                df = process_dataframe_complete(df)
                
                if df is None:
                    return
            
            st.success("✅ Archivo procesado exitosamente, puedes revisar tu análisis!!!")
            
            # Mostrar estadísticas básicas (EXACTO como en el Colab)
            st.markdown('<div class="section-header">📊 Estadísticas Básicas</div>', unsafe_allow_html=True)
            
            # Cálculos exactos del Colab
            total_message = df.shape[0]
            media_message = df[df['Message'] == '<Multimedia omitido>'].shape[0]
            del_message = df[df['Message'] == 'Se eliminó este mensaje'].shape[0]
            links = np.sum(df.URL_count)
            
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{total_message}</h3>
                    <p>Total Mensajes</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{media_message}</h3>
                    <p>Multimedia</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{del_message}</h3>
                    <p>Eliminados</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{round((media_message/total_message)*100, 2)}%</h3>
                    <p>% Multimedia</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col5:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{int(links)}</h3>
                    <p>Enlaces (LLLL)</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Pestañas con TODOS los análisis del Colab
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "👥 Análisis Miembros", 
                "📅 Actividad Temporal", 
                "😊 Emojis & Multimedia",
                "💬 Nubes de Palabras", 
                "📊 Estadísticas Completas"
            ])
            
            with tab1:
                st.markdown('<div class="section-header">👥 Análisis de Miembros: </div>', unsafe_allow_html=True)
                
                # 1. Top 10 miembros
                st.subheader("📈 Top de los 10 miembros con mayor cantidad de mensajes")
                Topper = df['Author'].value_counts().head(10)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                bars = ax.bar(Topper.index, Topper.values, color='#32CD32')
                for a, b in enumerate(Topper.values):
                    ax.text(a-0.15, b+8, str(b), color='black', fontsize=10)
                ax.set_xticklabels(Topper.index, rotation=0, size=10)
                ax.set_yticks([])
                ax.set_title('Top de los 10 miembros con mayor cantidad de mensajes', fontsize=13, fontweight='bold')
                st.pyplot(fig)
                plt.close()
                
                # Estadísticas detalladas
                st.subheader("📋 Estadísticas Detalladas por Miembro")
                stat_df = generate_member_stats(df)
                st.dataframe(stat_df, use_container_width=True)
                
                # Información adicional
                st.info(f"📊 Total de usuarios únicos: {len(df['Author'].unique())}")
            
            with tab2:
                st.markdown('<div class="section-header">📅 Actividad Temporal</div>', unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Actividad por día
                    st.subheader("📅 Actividad del chat por dia")
                    active_day = df['Day'].value_counts()
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    bars = ax.bar(active_day.index, active_day.values, color='#32CD32')
                    for a, b in enumerate(active_day.values):
                        ax.text(a-0.12, b+10, str(b), color='black', fontsize=10)
                    ax.set_xticklabels(active_day.index, rotation=0, size=10)
                    ax.set_yticks([])
                    ax.set_title('Actividad del chat por dia', fontsize=13, fontweight='bold')
                    st.pyplot(fig)
                    plt.close()
                
                with col2:
                    # Mensajes por año
                    st.subheader("📆 Mensajes por Año")
                    TopYear = df['Year'].value_counts()
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    bars = ax.bar(TopYear.index.astype(str), TopYear.values, color='#32CD32')
                    for a, b in enumerate(TopYear.values):
                        ax.text(a-0.04, b+15, str(b), color='black', fontsize=10)
                    ax.set_xticklabels(TopYear.index, rotation=0, size=10)
                    ax.set_yticks([])
                    ax.set_title('Mensajes por Año', fontsize=13, fontweight='bold')
                    st.pyplot(fig)
                    plt.close()
                
                # Mensajes por mes
                st.subheader("🗓️ Mensajes por Mes")
                TopMonth = df['Month'].value_counts()
                
                fig, ax = plt.subplots(figsize=(12, 6))
                bars = ax.bar(TopMonth.index, TopMonth.values, color='#32CD32')
                for a, b in enumerate(TopMonth.values):
                    ax.text(a-0.12, b+15, str(b), color='black', fontsize=10)
                ax.set_xticklabels(TopMonth.index, rotation=0, size=10)
                ax.set_yticks([])
                ax.set_title('Mensajes por Mes', fontsize=13, fontweight='bold')
                st.pyplot(fig)
                plt.close()
                
                # Horas más activa
                if 'Time' in df.columns:
                    st.subheader("⏰ Las 10 Horas mas activas del grupo")
                    try:
                        df['Hour'] = df['Time'].apply(lambda a: a.split(':')[0] if isinstance(a, str) and ':' in str(a) else '0')
                        TimeHours = df['Hour'].value_counts().head(10)
                        
                        fig, ax = plt.subplots(figsize=(12, 6))
                        bars = ax.bar(TimeHours.index, TimeHours.values, color='#32CD32')
                        for a, b in enumerate(TimeHours.values):
                            ax.text(a-0.15, b+7, str(b), color='black', fontsize=10)
                        ax.set_xticklabels(TimeHours.index, rotation=0, size=10)
                        ax.set_yticks([])
                        ax.set_title('Las 10 Horas mas activas del grupo', fontsize=13, fontweight='bold')
                        st.pyplot(fig)
                        plt.close()
                    except Exception as e:
                        st.warning(f"⚠️ No se pudieron procesar las horas: {e}")
                
                # Línea temporal
                st.subheader("📈 Linea Temporal de mensajes por mes")
                try:
                    TimeLine = df.groupby(['Year', 'Num_Month', 'Month']).count()['Message'].reset_index()
                    month_timeline = []
                    for i in range(TimeLine.shape[0]):
                        month_timeline.append(TimeLine['Month'].iloc[i] + '-' + str(TimeLine['Year'].iloc[i]))
                    TimeLine['Time'] = month_timeline
                    
                    fig, ax = plt.subplots(figsize=(14, 6))
                    ax.plot(TimeLine['Time'], TimeLine['Message'], color='#32CD32', linewidth=2, marker='o')
                    ax.set_xticklabels(TimeLine['Time'], rotation=0, size=6)
                    ax.set_title('Linea Temporal de mensajes por mes', fontsize=13, fontweight='bold')
                    st.pyplot(fig)
                    plt.close()
                except Exception as e:
                    st.warning(f"⚠️ Error en línea temporal: {e}")
            
            with tab3:
                st.markdown('<div class="section-header">😊 Emojis & Multimedia</div>', unsafe_allow_html=True)
                
                # Análisis de emojis 
                st.subheader("😊 Gráfico de los 10 emojis utilizados en el chat")
                emojis = []
                no_emojis = ['🏻', '🏼', '🏽', '🏾', '🏿', '🪄', '🪛']
                
                for i in df['Message']:
                    my_str = str(i)
                    for j in my_str:
                        if j in emoji.EMOJI_DATA:
                            if j not in no_emojis:
                                emojis.append(j)
                
                if emojis:
                    emo = pd.Series(emojis)
                    TopEmoji = emo.value_counts().head(10)
                    emoji_dict = dict(TopEmoji)
                    emoji_dict = sorted(emoji_dict.items(), key=lambda x: x[1], reverse=True)
                    emoji_df = pd.DataFrame(emoji_dict, columns=['emoji', 'count'])
                    
                    fig = px.pie(emoji_df, values='count', names='emoji')
                    fig.update_traces(textposition='inside', textinfo='percent+label')
                    fig.update_layout(title='Gráfico de los 10 emojis utilizados en el chat')
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("ℹ️ No se encontraron emojis en el chat")
                
                # Multimedia por miembro 
                st.subheader("📷 Top de los 10 Miembros con mayor envio de multimedia")
                MediaValue = df[df['Message'] == '<Multimedia omitido>']
                
                if not MediaValue.empty:
                    MediaTopper = MediaValue['Author'].value_counts().head(10)
                    
                    fig, ax = plt.subplots(figsize=(12, 6))
                    bars = ax.bar(MediaTopper.index, MediaTopper.values, color='#32CD32')
                    for a, b in enumerate(MediaTopper.values):
                        ax.text(a-0.16, b+0.1, str(b), color='black', fontsize=10)
                    ax.set_xticklabels(MediaTopper.index, rotation=0, size=10)
                    ax.set_yticks([])
                    ax.set_title('Top de los 10 Miembros con mayor envio de multimedia', fontsize=13, fontweight='bold')
                    st.pyplot(fig)
                    plt.close()
                else:
                    st.info("ℹ️ No se encontraron mensajes multimedia en el chat")
            
            with tab4:
                st.markdown('<div class="section-header">💬 Nubes de Palabras (utils.ipynb)</div>', unsafe_allow_html=True)
                
                # Nube de palabras del chat completo (función exacta de utils.ipynb)
                st.subheader("☁️ Nube de Palabras - Chat Completo")
                try:
                    wordcloud_fig = Nube_Words(df, "Chat Completo")
                    st.pyplot(wordcloud_fig)
                    plt.close()
                except Exception as e:
                    st.error(f"❌ Error generando nube de palabras: {e}")
                
                # Nubes por fechas más activas 
                st.subheader("📅 Nubes de Palabras - Días Más Activos")
                TopDate = df['Date'].value_counts().head(5)
                
                for i in range(min(3, len(TopDate))):
                    fecha = TopDate.index[i]
                    mensajes = TopDate.iloc[i]
                    
                    with st.expander(f"📅 {fecha} - {mensajes} mensajes", expanded=(i==0)):
                        df_fecha = df[df['Date'] == fecha]
                        if not df_fecha.empty:
                            try:
                                wordcloud_fecha = Nube_Words(df_fecha, fecha)
                                st.pyplot(wordcloud_fecha)
                                plt.close()
                            except Exception as e:
                                st.error(f"❌ Error: {e}")
                        else:
                            st.info("ℹ️ No hay mensajes para este día")
                
                # Nube por usuario seleccionado
                st.subheader("👤 Nube de Palabras por Usuario")
                selected_user = st.selectbox("Selecciona un usuario:", df['Author'].unique())
                
                if selected_user:
                    user_df = df[df['Author'] == selected_user]
                    try:
                        user_wordcloud = Nube_Words(user_df, selected_user)
                        st.pyplot(user_wordcloud)
                        plt.close()
                    except Exception as e:
                        st.error(f"❌ Error: {e}")
            
            with tab5:
                st.markdown('<div class="section-header">📊 Estadísticas Completas</div>', unsafe_allow_html=True)
                
                # Fechas más activas 
                st.subheader("📅 Los 10 Dias mas activos del chat")
                TopDate = df['Date'].value_counts().head(10)
                
                fig, ax = plt.subplots(figsize=(14, 6))
                bars = ax.bar(TopDate.index, TopDate.values, color='#32CD32')
                for a, b in enumerate(TopDate.values):
                    ax.text(a-0.15, b+2, str(b), color='black', fontsize=10)
                ax.set_xticklabels(TopDate.index, rotation=5, size=8)
                ax.set_yticks([])
                ax.set_title('Los 10 Dias mas activos del chat', fontsize=13, fontweight='bold')
                st.pyplot(fig)
                plt.close()
                
                # Resumen estadístico
                st.subheader("📈 Resumen Estadístico")
                st.write("**Estadísticas de texto:**")
                st.write(f"- Promedio palabras por mensaje: {df['Words'].mean():.2f}")
                st.write(f"- Promedio caracteres por mensaje: {df['Letters'].mean():.2f}")
                st.write(f"- Total palabras: {df['Words'].sum():,}")
                st.write(f"- Total caracteres: {df['Letters'].sum():,}")
                st.write(f"- Total enlaces encontrados: {df['URL_count'].sum()}")
        
        except Exception as e:
            st.error(f"❌ Error procesando el archivo: {str(e)}")
            st.info("🔍 Asegúrate de que el archivo sea un chat exportado de WhatsApp en formato .txt")
    
    else:
        # Página de inicio
        st.markdown("""
        ### 🚀 ¡Bienvenido al Analizador de Chats de WhatsApp!
        
        **Esta aplicación implementa código utilizado en el Examen del segundo parcial de Inteligencia Artificial.**
        
        #### 📋 Análisis incluidos:
        
        1. **👥 Top de los 10 miembros con mayor cantidad de mensajes**
        2. **📅 Actividad del chat por día**  
        3. **😊 Gráfico de los 10 emojis utilizados en el chat**
        4. **📷 Top de los 10 Miembros con mayor envío de multimedia**
        5. **📊 Los 10 días más activos del chat**
        6. **📆 Mensajes por año**
        7. **🗓️ Mensajes por mes**
        8. **⏰ Las 10 horas más activas del grupo**
        9. **📈 Línea temporal de mensajes por mes**
        10. **☁️ Nubes de palabras (usando utils.ipynb)**
        11. **📋 Estadísticas detalladas por miembro**
        12. **🔍 DataFrame procesado completo**
        
        #### 📤 Para comenzar:
        1. **Exporta tu chat de WhatsApp:**
           - Abre el chat → Menú (⋮) → Más → Exportar chat → Sin multimedia
        2. **Sube el archivo .txt en la barra lateral**
        3. **Explora todos los análisis en las pestañas**
        
        ---
        
        **📚 Código basado en el Examen 2do Parcial - Inteligencia Artificial**  
        **🎓 Universidad de las Fuerzas Armadas ESPE**
        """)

if __name__ == "__main__":
    main()