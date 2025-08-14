<map version="freeplane 1.12.1">
<!--To view this file, download free mind mapping software Freeplane from https://www.freeplane.org -->
<node TEXT="Transolver + 3D PointNet" FOLDED="false" ID="ID_696401721" CREATED="1610381621824" MODIFIED="1755142419658" STYLE="oval" VGAP_QUANTITY="2 pt" COMMON_HGAP_QUANTITY="14 pt">
<font SIZE="18"/>
<hook NAME="MapStyle">
    <properties edgeColorConfiguration="#808080ff,#ff0000ff,#0000ffff,#00ff00ff,#ff00ffff,#00ffffff,#7c0000ff,#00007cff,#007c00ff,#7c007cff,#007c7cff,#7c7c00ff" show_icon_for_attributes="true" auto_compact_layout="true" show_tags="UNDER_NODES" associatedTemplateLocation="template:/common.mm" show_note_icons="true" followedTemplateLocation="template:/standard-1.6.mm" followedMapLastTime="1748784985000" fit_to_viewport="false" show_icons="BESIDE_NODES" showTagCategories="false"/>
    <tags category_separator="::"/>

<map_styles>
<stylenode LOCALIZED_TEXT="styles.root_node" STYLE="oval" UNIFORM_SHAPE="true" VGAP_QUANTITY="24 pt">
<font SIZE="24"/>
<stylenode LOCALIZED_TEXT="styles.predefined" POSITION="bottom_or_right" STYLE="bubble">
<stylenode LOCALIZED_TEXT="default" ID="ID_271890427" ICON_SIZE="12 pt" FORMAT_AS_HYPERLINK="false" COLOR="#000000" STYLE="fork" NUMBERED="false" FORMAT="STANDARD_FORMAT" TEXT_ALIGN="DEFAULT" TEXT_WRITING_DIRECTION="LEFT_TO_RIGHT" VGAP_QUANTITY="2 pt" COMMON_HGAP_QUANTITY="14 pt" CHILD_NODES_LAYOUT="AUTO" BORDER_WIDTH_LIKE_EDGE="false" BORDER_WIDTH="1 px" BORDER_COLOR_LIKE_EDGE="true" BORDER_COLOR="#808080" BORDER_DASH_LIKE_EDGE="false" BORDER_DASH="SOLID">
<arrowlink SHAPE="CUBIC_CURVE" COLOR="#000000" WIDTH="2" TRANSPARENCY="200" DASH="" FONT_SIZE="9" FONT_FAMILY="SansSerif" DESTINATION="ID_271890427" STARTARROW="NONE" ENDARROW="DEFAULT"/>
<font NAME="Times New Roman" SIZE="10" BOLD="false" STRIKETHROUGH="false" ITALIC="false"/>
<richcontent TYPE="DETAILS" CONTENT-TYPE="plain/auto"/>
<richcontent TYPE="NOTE" CONTENT-TYPE="plain/auto"/>
<edge COLOR="#808080"/>
</stylenode>
<stylenode LOCALIZED_TEXT="defaultstyle.details"/>
<stylenode LOCALIZED_TEXT="defaultstyle.tags">
<font SIZE="10"/>
</stylenode>
<stylenode LOCALIZED_TEXT="defaultstyle.attributes">
<font SIZE="9"/>
</stylenode>
<stylenode LOCALIZED_TEXT="defaultstyle.note" COLOR="#000000" BACKGROUND_COLOR="#ffffff" TEXT_ALIGN="LEFT"/>
<stylenode LOCALIZED_TEXT="defaultstyle.floating">
<edge STYLE="hide_edge"/>
<cloud COLOR="#f0f0f0" SHAPE="ROUND_RECT"/>
</stylenode>
<stylenode LOCALIZED_TEXT="defaultstyle.selection" BACKGROUND_COLOR="#afd3f7" BORDER_COLOR_LIKE_EDGE="false" BORDER_COLOR="#afd3f7"/>
</stylenode>
<stylenode LOCALIZED_TEXT="styles.user-defined" POSITION="bottom_or_right" STYLE="bubble">
<stylenode LOCALIZED_TEXT="styles.topic" COLOR="#18898b" STYLE="fork">
<font NAME="Liberation Sans" SIZE="10" BOLD="true"/>
</stylenode>
<stylenode LOCALIZED_TEXT="styles.subtopic" COLOR="#cc3300" STYLE="fork">
<font NAME="Liberation Sans" SIZE="10" BOLD="true"/>
</stylenode>
<stylenode LOCALIZED_TEXT="styles.subsubtopic" COLOR="#669900">
<font NAME="Liberation Sans" SIZE="10" BOLD="true"/>
</stylenode>
<stylenode LOCALIZED_TEXT="styles.important" ID="ID_67550811">
<icon BUILTIN="yes"/>
<arrowlink COLOR="#003399" TRANSPARENCY="255" DESTINATION="ID_67550811"/>
</stylenode>
<stylenode LOCALIZED_TEXT="styles.flower" COLOR="#ffffff" BACKGROUND_COLOR="#255aba" STYLE="oval" TEXT_ALIGN="CENTER" BORDER_WIDTH_LIKE_EDGE="false" BORDER_WIDTH="22 pt" BORDER_COLOR_LIKE_EDGE="false" BORDER_COLOR="#f9d71c" BORDER_DASH_LIKE_EDGE="false" BORDER_DASH="CLOSE_DOTS" MAX_WIDTH="6 cm" MIN_WIDTH="3 cm"/>
</stylenode>
<stylenode LOCALIZED_TEXT="styles.AutomaticLayout" POSITION="bottom_or_right" STYLE="bubble">
<stylenode LOCALIZED_TEXT="AutomaticLayout.level.root" COLOR="#000000" STYLE="oval" SHAPE_HORIZONTAL_MARGIN="10 pt" SHAPE_VERTICAL_MARGIN="10 pt">
<font SIZE="18"/>
</stylenode>
<stylenode LOCALIZED_TEXT="AutomaticLayout.level,1" COLOR="#0033ff">
<font SIZE="16"/>
</stylenode>
<stylenode LOCALIZED_TEXT="AutomaticLayout.level,2" COLOR="#00b439">
<font SIZE="14"/>
</stylenode>
<stylenode LOCALIZED_TEXT="AutomaticLayout.level,3" COLOR="#990000">
<font SIZE="12"/>
</stylenode>
<stylenode LOCALIZED_TEXT="AutomaticLayout.level,4" COLOR="#111111">
<font SIZE="10"/>
</stylenode>
<stylenode LOCALIZED_TEXT="AutomaticLayout.level,5"/>
<stylenode LOCALIZED_TEXT="AutomaticLayout.level,6"/>
<stylenode LOCALIZED_TEXT="AutomaticLayout.level,7"/>
<stylenode LOCALIZED_TEXT="AutomaticLayout.level,8"/>
<stylenode LOCALIZED_TEXT="AutomaticLayout.level,9"/>
<stylenode LOCALIZED_TEXT="AutomaticLayout.level,10"/>
<stylenode LOCALIZED_TEXT="AutomaticLayout.level,11"/>
</stylenode>
</stylenode>
</map_styles>
</hook>
<hook NAME="AutomaticEdgeColor" COUNTER="13" RULE="ON_BRANCH_CREATION"/>
<hook NAME="accessories/plugins/AutomaticLayout.properties" VALUE="ALL"/>
<node TEXT="Capture local geometry" POSITION="bottom_or_right" ID="ID_1392340545" CREATED="1755142129719" MODIFIED="1755143457104" HGAP_QUANTITY="53.68504 pt" VSHIFT_QUANTITY="-68.59843 pt">
<edge COLOR="#7c007c"/>
<node TEXT="knn()" ID="ID_1134716685" CREATED="1755142139398" MODIFIED="1755142205699"><richcontent TYPE="DETAILS">
<html>
  <head>
    
  </head>
  <body>
    <p>
      获取当前点临近40个点的位置信息
    </p>
  </body>
</html></richcontent>
</node>
<node TEXT="get_graph_feature()" ID="ID_801754831" CREATED="1755142138481" MODIFIED="1755142453344" VSHIFT_QUANTITY="19.84252 pt"><richcontent TYPE="DETAILS">
<html>
  <head>
    
  </head>
  <body>
    <p>
      根据knn() 构造新的数据结构
    </p>
  </body>
</html></richcontent>
<node TEXT="[B, num_points, feature_dim]&#xa;-&gt;&#xa;[B, num_points, 2*f_dim, K]" ID="ID_1713861036" CREATED="1755142238612" MODIFIED="1755142353928"/>
</node>
</node>
<node TEXT="模型修改" POSITION="bottom_or_right" ID="ID_859301058" CREATED="1755142489520" MODIFIED="1755142802004" HGAP_QUANTITY="46.31496 pt" VSHIFT_QUANTITY="26.64567 pt">
<edge COLOR="#007c7c"/>
<node TEXT="Transolver: [B, num_points, f_dim]" ID="ID_238390928" CREATED="1755142535721" MODIFIED="1755142576248"/>
<node TEXT="Current: [B, num_points, 2*f_dim, K]" ID="ID_1158551020" CREATED="1755142578441" MODIFIED="1755142605991">
<node TEXT="利用max pooling这个操作消去维度K，最后进入Transolver 模型" ID="ID_667732309" CREATED="1755142607604" MODIFIED="1755142678223"><richcontent TYPE="DETAILS">
<html>
  <head>
    
  </head>
  <body>
    <p>
      作为前处理
    </p>
  </body>
</html></richcontent>
</node>
</node>
</node>
<node TEXT="Methon 2" POSITION="top_or_left" ID="ID_88295939" CREATED="1755142804637" MODIFIED="1755142814030">
<edge COLOR="#ff0000"/>
<node TEXT="设置一个3维的参考网格" ID="ID_1881222712" CREATED="1755142818036" MODIFIED="1755142849601"/>
<node TEXT="将物理单元节点的坐标与参考单元节点的坐标的差值作为局部结构" ID="ID_1262981702" CREATED="1755142850543" MODIFIED="1755142900400"/>
<node TEXT="[B, Phy_points, ref_points, ref_idx]" ID="ID_1305708942" CREATED="1755143093535" MODIFIED="1755143411757"/>
</node>
</node>
</map>
