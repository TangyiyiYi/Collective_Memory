if (duse == "all") {
  dfcg3=dfcg
  txuse="50 glitch version + 25 normal version"
} else if (duse=="v2"){
  dfcg3=dfcg%>%filter(codeversion>10)
  txuse="25 normal version"
} else if(duse =="v1"){
  dfcg3=dfcg%>%filter(codeversion==10)
  txuse="50 glitch version"
}

# dfcg1 = dfcg%>%filter(word_response_finalt!="none")
dfnew4=dfcg3 %>% filter(task%in% c("finalt_response","pretest_response"))%>% 
  mutate(listnumber_1to10=as.numeric(listnumber_1to10))%>%
  mutate(rt=as.numeric(rt))%>%
  filter(rt<30000,correct_all)%>%
  group_by(PROLIFIC_PID,task,listnumber_1to10,condition,wordcondi)%>%
  summarise(accuracy1=mean(rt))%>%
  group_by(task,listnumber_1to10,condition,wordcondi)%>%
  summarise(accuracy=mean(accuracy1),sd=sd(accuracy1),se=sd/sqrt(n()))%>%
  mutate(`by-initial-test` = accuracy,se_initial=se)%>%
  select(task,listnumber_1to10,condition,wordcondi,`by-initial-test`,se_initial)
# dfnew4

dfnew42= dfcg3 %>% filter(task%in% c("finalt_response"))%>% 
  group_by(PROLIFIC_PID)%>%
  mutate(listnumber_1to10 = listgroup_finaltest) %>%
  mutate(rt=as.numeric(rt))%>%
  filter(rt<30000,correct_all)%>%
  group_by(PROLIFIC_PID,task,listnumber_1to10,condition,wordcondi)%>%
  summarise(accuracy1=mean(rt))%>%
  group_by(task,listnumber_1to10,condition,wordcondi)%>%
  summarise(accuracy=mean(accuracy1),sd=sd(accuracy1),se=sd/sqrt(n()))%>%
  mutate(`by-final-test`=accuracy,se_final=se)%>%
  select(task,listnumber_1to10,condition,wordcondi,`by-final-test`,se_final)
# dfnew42

dftp0 = dfnew4%>%left_join(dfnew42,by=c("task","listnumber_1to10","condition","wordcondi"))

dftp1=dftp0%>%pivot_longer(cols=c(`by-final-test`,`by-initial-test`),
                           names_to = "name",values_to = "accuracy")%>%filter(complete.cases(accuracy))%>%
  select(-c(se_initial,se_final))
# dftp1

dftp2 = dftp0%>%
  pivot_longer(cols=starts_with("se"),names_to = "name",values_to = "se")%>%
  mutate(name=case_when(name=="se_initial"~"by-initial-test",
                        name=="se_final"~"by-final-test"))%>%filter(complete.cases(se))%>%
  select(-starts_with("by"))
# dftp2

dfnew44=dftp2%>%left_join(dftp1,by=c("task","listnumber_1to10","condition","wordcondi","name"))%>%
  mutate(task=case_when(task=="finalt_response"~paste("finalT",name,sep="-"),
                        task=="pretest_response"~paste("initialT",name,sep="-")))%>%
  mutate(wordcondi=as.factor(wordcondi))

dfnew44

dfnew4avg=dfcg3%>%filter(task%in% c("pretest_response"))%>% 
  mutate(listnumber_1to10=as.numeric(listnumber_1to10))%>%
  mutate(rt=as.numeric(rt))%>%
  filter(rt<30000,correct_all)%>%
  group_by(PROLIFIC_PID,task,listnumber_1to10,condition,wordcondi)%>%
  summarise(accuracy1=mean(rt))%>%
  group_by(task,listnumber_1to10,wordcondi)%>%
  summarise(accuracy=mean(accuracy1),sd=sd(accuracy1),se=sd/sqrt(n()))%>%
  # mutate(accuracy=mean(accuracy1),sd=sd(accuracy1),se=sd/sqrt(n()))%>%
  # group_by(task,listnumber_1to10,wordcondi,condition)%>%
  # summarise(accuracy=mean(accuracy),sd=mean(sd),se=mean(se))%>%
  mutate(task="initialT-by-initial-test",condition="Averaged")

dflabel4= data.frame(task=rep(c("finalT-by-final-test","finalT-by-initial-test"),5),
                     label1=paste(c("first saw","last saw","first saw","first saw","first-saw","first-last-saw","first-saw","last-first-saw","first-saw","random"),"in final"),
                     label2=c("list8,7...","list1,2...","list1,2...","list1,2..","list1,8...","list1,2...","list8,1...","list1,2...","random list","list1,2"),
                     condition=rep(c("Backward","Forward","FRandom","BRandom", "Random"),each=2))
dflabel4
ggplot(data=dfnew44,aes(listnumber_1to10,accuracy,group=interaction(task,wordcondi,condition)))+
  # geom_line(aes(shape=wordcondi,color=wordcondi))+
  geom_point(data=dfnew44%>%filter(task!="initialT-by-initial-test"),aes(shape=wordcondi,color=wordcondi))+
  geom_line(data=dfnew44%>%filter(task!="initialT-by-initial-test"),aes(color=wordcondi,linetype=wordcondi)) +
  geom_line(data=dfnew4avg,aes(color=wordcondi,linetype=wordcondi))+
  geom_line(data=dfnew44%>%filter(task=="initialT-by-initial-test"),aes(linetype=wordcondi),color="black",alpha=0.3) +
  geom_errorbar(data=dfnew4avg,aes(ymin=accuracy-se,ymax=accuracy+se,color=wordcondi),alpha=0.8,width=0.2)+
  geom_ribbon(data=dfnew4avg,aes(ymin=accuracy-se,ymax=accuracy+se,fill=wordcondi),alpha=0.1)+
  geom_ribbon(data=dfnew44%>%filter(task!="initialT-by-initial-test"),aes(ymin=accuracy-se,ymax=accuracy+se,fill=wordcondi),alpha=0.1)+
  labs(x="List number (by final/initial test position, see column panel label)",
       y="correct RT",
       title=paste(txuse,"
       Correct RT plot; Row panels: Averaged data and 5 conditions. Column panels: 
       left:Final Test result by final test position, 
       middle: Final test result by inital test position,
       right: Inital test result by inital test position"))+
  # theme_light()+
  theme(text=element_text(size=15),plot.title=element_text(size=13))+
  facet_grid(condition~task)+
  scale_x_continuous(breaks=1:8,labels=1:8)+
  # annotate(data=dfnew44%>%filter(task=="finalT-by-final-test"),"text", x = 1, y = 0.3, label = "inital", color = "red")+
  geom_text(data = dflabel4, aes(x = -Inf, y =-Inf, hjust=-0.05, vjust=-0.5, label = label1),alpha=0.5,inherit.aes = FALSE)+
  geom_text(data = dflabel4, aes(x = -Inf, y =Inf, hjust=-0.1, vjust=1.5, label = label2),alpha=0.5,inherit.aes = FALSE)
# ylim(0,0.6)
levels(dfnew44$task%>%as.factor())