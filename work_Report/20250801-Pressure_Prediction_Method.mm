<map version="freeplane 1.12.1">
<!--To view this file, download free mind mapping software Freeplane from https://www.freeplane.org -->
<node TEXT="Pressure Prediction" FOLDED="false" ID="ID_696401721" CREATED="1610381621824" MODIFIED="1754028925251" STYLE="oval" VGAP_QUANTITY="2 pt" COMMON_HGAP_QUANTITY="14 pt">
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
<hook NAME="AutomaticEdgeColor" COUNTER="12" RULE="ON_BRANCH_CREATION"/>
<hook NAME="accessories/plugins/AutomaticLayout.properties" VALUE="ALL"/>
<node TEXT="Transolver" POSITION="bottom_or_right" ID="ID_1959573932" CREATED="1754028419138" MODIFIED="1754028934130" HGAP_QUANTITY="38.37795 pt" VSHIFT_QUANTITY="-20.97638 pt" VGAP_QUANTITY="2 pt" COMMON_HGAP_QUANTITY="14 pt">
<edge COLOR="#007c00"/>
<richcontent TYPE="DETAILS">
<html>
  <head>
    
  </head>
  <body>
    <p>
      方案一
    </p>
  </body>
</html></richcontent>
<node TEXT="利用Transolver模型对MIT所提供的压强场点云文件进行预测，三维切片" ID="ID_781122816" CREATED="1754028479707" MODIFIED="1754029161694"><richcontent TYPE="DETAILS">
<html>
  <head>
    
  </head>
  <body>
    <p>
      由Wu2024中提到2D模型的点云文件，利用Transolver模型进行预测
    </p>
  </body>
</html></richcontent>
</node>
</node>
<node TEXT="DrivAriNet" POSITION="bottom_or_right" ID="ID_520428919" CREATED="1754028910819" MODIFIED="1754029126494" HGAP_QUANTITY="21.93701 pt" VSHIFT_QUANTITY="48.18898 pt" VGAP_QUANTITY="2 pt" COMMON_HGAP_QUANTITY="14 pt">
<edge COLOR="#7c007c"/>
<richcontent TYPE="DETAILS">
<html>
  <head>
    
  </head>
  <body>
    <p>
      方案二
    </p>
  </body>
</html></richcontent>
<node TEXT="利用Huang2023文章中的方法，修改DrivAriNet模型，对压强场重新预测" ID="ID_1413828799" CREATED="1754028966826" MODIFIED="1754029130328" VSHIFT_QUANTITY="-3.40157 pt"/>
<node TEXT="\latex $(x_i, x_i-x_j, n_k)$" ID="ID_1463594458" CREATED="1754029069993" MODIFIED="1754029335717" HGAP_QUANTITY="18.53543 pt" VSHIFT_QUANTITY="-15.87402 pt"/>
<node TEXT="Use Coarse Subset" ID="ID_714839722" CREATED="1752289067937" MODIFIED="1752289080719">
<node TEXT="Multiple Sampling" ID="ID_1342272601" CREATED="1752289080724" MODIFIED="1752289088105"/>
<node TEXT="Outlier Filter" ID="ID_1877715637" CREATED="1752289088644" MODIFIED="1752289104614"/>
</node>
</node>
<node TEXT="后续想法" POSITION="bottom_or_right" ID="ID_628666576" CREATED="1754029493844" MODIFIED="1754029515955" HGAP_QUANTITY="25.90551 pt" VSHIFT_QUANTITY="42.51969 pt">
<edge COLOR="#ff0000"/>
<node TEXT="多种物理场预测" ID="ID_1333448560" CREATED="1754029515961" MODIFIED="1754029661218">
<node TEXT="调研文献，根据MIT那个团队提供的数据，把视角放到多种物理场的预测" ID="ID_1053827474" CREATED="1754029662226" MODIFIED="1754029663775"/>
</node>
<node TEXT="分析各类模型的预测效果" ID="ID_647533997" CREATED="1754029645267" MODIFIED="1754029715752">
<node TEXT="误差的概率分布" ID="ID_1416854555" CREATED="1754029670546" MODIFIED="1754029766985"/>
<node TEXT="扰动干扰对模型的影响" ID="ID_1627224877" CREATED="1754029730524" MODIFIED="1754029758801"/>
<node TEXT="看看是否可以将可预测部分和不可预测部分分解出来。" ID="ID_1002675362" CREATED="1754029768147" MODIFIED="1754029769611"/>
</node>
</node>
<node TEXT="目前的困惑" POSITION="top_or_left" ID="ID_1075346026" CREATED="1754029359844" MODIFIED="1754029783586" VGAP_QUANTITY="2 pt" COMMON_HGAP_QUANTITY="14 pt">
<edge COLOR="#007c7c"/>
<node TEXT="在理解模型的时候，涉及高阶张量的运算，尤其是对于其物理意义的理解上很勉强，是否值得花时间去自学高等线性代数" ID="ID_1074689697" CREATED="1754029371927" MODIFIED="1754029796379" HGAP_QUANTITY="15.70079 pt"/>
</node>
</node>
</map>
